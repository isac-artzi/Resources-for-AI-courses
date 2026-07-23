"""
FastAPI Backend for Deep Learning Image Classification

This application provides a REST API for image classification using TensorFlow.
It's designed to be deployed on Render.com and serve a React frontend on Vercel.

Author: Your Name
Date: 2026
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Dict
import io
from PIL import Image
import numpy as np
import logging

# Import our model handler
from model import ModelHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Deep Learning Image Classification API",
    description="API for classifying images using deep learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS - Allow requests from frontend
# Update these URLs based on your deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",          # Local React development
        "http://localhost:5173",          # Local Vite development
        "https://*.vercel.app",           # Vercel preview deployments
        "https://your-app.vercel.app",    # Replace with your actual Vercel URL
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Initialize the model handler
model_handler = ModelHandler()

# Startup event - Load model when server starts
@app.on_event("startup")
async def startup_event():
    """
    Load the deep learning model when the server starts.
    This ensures the model is ready before handling requests.
    """
    logger.info("Loading deep learning model...")
    try:
        model_handler.load_model()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Continue startup even if model fails to load
        # This allows health checks to work


# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint - Welcome message and API status

    Returns:
        dict: Welcome message and status
    """
    return {
        "message": "Welcome to Deep Learning API",
        "status": "active",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "predict": "/predict",
            "health": "/health"
        }
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring

    Returns:
        dict: Health status of the API
    """
    return {
        "status": "healthy",
        "model_loaded": model_handler.model is not None
    }


# Prediction endpoint
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Upload an image and get classification predictions

    Args:
        file (UploadFile): The image file to classify

    Returns:
        JSONResponse: Predictions with class names and confidence scores

    Raises:
        HTTPException: If there's an error processing the image
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPEG, PNG, etc.)"
            )

        # Read image file
        logger.info(f"Processing image: {file.filename}")
        contents = await file.read()

        # Convert to PIL Image
        image = Image.open(io.BytesIO(contents))

        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Get predictions from model
        predictions = model_handler.predict(image)

        logger.info(f"Predictions generated for {file.filename}")

        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "predictions": predictions
        })

    except HTTPException as he:
        # Re-raise HTTP exceptions
        raise he
    except Exception as e:
        # Log error and return 500
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


# Batch prediction endpoint (optional - for multiple images)
@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Upload multiple images and get classification predictions for each

    Args:
        files (List[UploadFile]): List of image files to classify

    Returns:
        JSONResponse: List of predictions for each image

    Raises:
        HTTPException: If there's an error processing the images
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images allowed per batch"
        )

    results = []

    for file in files:
        try:
            # Validate file type
            if not file.content_type.startswith('image/'):
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "File must be an image"
                })
                continue

            # Read and process image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))

            if image.mode == 'RGBA':
                image = image.convert('RGB')

            # Get predictions
            predictions = model_handler.predict(image)

            results.append({
                "filename": file.filename,
                "success": True,
                "predictions": predictions
            })

        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })

    return JSONResponse(content={
        "success": True,
        "total_images": len(files),
        "results": results
    })


# Model info endpoint
@app.get("/model/info")
async def model_info():
    """
    Get information about the loaded model

    Returns:
        dict: Model information including name and version
    """
    return model_handler.get_model_info()


# Main entry point for local development
if __name__ == "__main__":
    # Run the server
    # For production, use: uvicorn main:app --host 0.0.0.0 --port $PORT
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (development only)
        log_level="info"
    )
