"""
FastAPI Backend for Deep Learning Image Classification
======================================================

This module serves as the main entry point for the FastAPI application.
It handles HTTP requests, processes images, and returns ML predictions.

Key Features:
- CORS middleware for cross-origin requests
- File upload handling with validation
- Image classification using MobileNetV2
- Error handling and logging
- Health check endpoints

Author: Cloud ML Deployment
"""

# Import required libraries
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from datetime import datetime
import logging
from typing import Dict, List
import io
from PIL import Image
import numpy as np

# Import our custom modules
from app.model import ImageClassifier
from app.utils import validate_image, format_predictions

# Configure logging for debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Deep Learning Image Classification API",
    description="API for classifying images using pre-trained MobileNetV2 model",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI documentation
    redoc_url="/redoc"  # ReDoc documentation
)

# ============================================================================
# CORS Configuration
# ============================================================================
# CORS (Cross-Origin Resource Sharing) allows frontend to make requests
# from a different domain/port. This is essential for development and production.

# List of allowed origins - UPDATE THESE with your actual URLs
allowed_origins = [
    "http://localhost:5173",  # Vite dev server (default)
    "http://localhost:3000",  # Alternative React dev port
    "http://localhost:5174",  # Alternative Vite port
    "https://*.vercel.app",   # Vercel preview deployments
    "https://your-app.vercel.app",  # Your production frontend URL
]

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,  # Allow cookies and authentication headers
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# ============================================================================
# Initialize ML Model
# ============================================================================
# Load the model when the application starts (singleton pattern)
# This happens once, not on every request, for better performance

logger.info("Loading ML model...")
try:
    classifier = ImageClassifier()
    logger.info("✓ Model loaded successfully")
except Exception as e:
    logger.error(f"✗ Failed to load model: {str(e)}")
    classifier = None

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root() -> Dict[str, str]:
    """
    Root endpoint - Basic health check

    Returns:
        dict: Basic API information

    Example:
        GET http://localhost:8000/
        Response: {"message": "FastAPI Deep Learning API", "status": "running"}
    """
    return {
        "message": "FastAPI Deep Learning API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check() -> Dict:
    """
    Detailed health check endpoint

    Returns:
        dict: Detailed health status including model availability

    Example:
        GET http://localhost:8000/health
        Response: {
            "status": "healthy",
            "model_loaded": true,
            "timestamp": "2024-01-15T10:30:00"
        }
    """
    return {
        "status": "healthy" if classifier else "degraded",
        "model_loaded": classifier is not None,
        "model_name": "MobileNetV2" if classifier else None,
        "timestamp": datetime.now().isoformat(),
        "framework": "TensorFlow/Keras"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """
    Main prediction endpoint - Classifies uploaded images

    This endpoint:
    1. Receives an image file from the client
    2. Validates the file type and size
    3. Preprocesses the image
    4. Runs inference using MobileNetV2
    5. Returns top-5 predictions with confidence scores

    Args:
        file (UploadFile): Image file uploaded by the client
                          Accepted formats: JPEG, PNG, JPG
                          Recommended size: < 10MB

    Returns:
        JSONResponse: Predictions with confidence scores

    Raises:
        HTTPException: 400 if file is invalid
        HTTPException: 500 if processing fails

    Example Request:
        POST http://localhost:8000/predict
        Content-Type: multipart/form-data
        Body: file=<image.jpg>

    Example Response:
        {
            "success": true,
            "predictions": [
                {"class": "golden_retriever", "confidence": 0.89},
                {"class": "Labrador_retriever", "confidence": 0.06},
                {"class": "dog", "confidence": 0.03}
            ],
            "processing_time": 0.234,
            "model": "MobileNetV2"
        }
    """

    # Start timing for performance monitoring
    start_time = time.time()

    # ========================================================================
    # Step 1: Check if model is loaded
    # ========================================================================
    if classifier is None:
        logger.error("Model not loaded - cannot process request")
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please try again later."
        )

    # ========================================================================
    # Step 2: Validate file upload
    # ========================================================================
    try:
        # Check file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Please upload an image."
            )

        # Validate image format
        validate_image(file.filename)

        logger.info(f"Processing image: {file.filename} ({file.content_type})")

    except ValueError as e:
        logger.warning(f"File validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    # ========================================================================
    # Step 3: Read and process image
    # ========================================================================
    try:
        # Read file contents into memory
        contents = await file.read()

        # Open image using PIL
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB if necessary (handles PNG with alpha channel, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.info(f"Converted image from {image.mode} to RGB")

        logger.info(f"Image loaded: {image.size} pixels")

    except Exception as e:
        logger.error(f"Failed to read image: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Could not read image file: {str(e)}"
        )

    # ========================================================================
    # Step 4: Run ML prediction
    # ========================================================================
    try:
        # Convert PIL Image to numpy array
        image_array = np.array(image)

        # Get predictions from the model
        predictions = classifier.predict(image_array)

        # Format predictions for response
        formatted_predictions = format_predictions(predictions, top_k=5)

        logger.info(f"Predictions generated: {formatted_predictions[0]['class']}")

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {str(e)}"
        )

    # ========================================================================
    # Step 5: Calculate processing time and return response
    # ========================================================================
    processing_time = time.time() - start_time

    response = {
        "success": True,
        "predictions": formatted_predictions,
        "processing_time": round(processing_time, 3),
        "model": "MobileNetV2",
        "image_size": image.size,
        "filename": file.filename
    }

    logger.info(f"Request completed in {processing_time:.3f}s")

    return JSONResponse(content=response)


# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Run when the application starts
    Useful for initializing resources, connections, etc.
    """
    logger.info("=" * 50)
    logger.info("FastAPI Deep Learning API Starting...")
    logger.info("=" * 50)
    if classifier:
        logger.info("✓ Model ready for predictions")
    else:
        logger.warning("✗ Model not loaded - predictions will fail")
    logger.info("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    """
    Run when the application shuts down
    Useful for cleanup, closing connections, etc.
    """
    logger.info("Shutting down FastAPI application...")
    # Add cleanup code here if needed (close DB connections, etc.)


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for uncaught errors
    Prevents the server from crashing and provides useful error messages
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if app.debug else "An unexpected error occurred"
        }
    )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # This block runs only when executing this file directly
    # For development: python -m app.main
    # For production: Use uvicorn directly

    import uvicorn

    # Run the application
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",  # Listen on all network interfaces
        port=8000,       # Default port
        reload=True,     # Auto-reload on code changes (development only!)
        log_level="info"
    )

    # Production command (use this for deployment):
    # uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 4
