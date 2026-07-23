"""
FastAPI Backend for Deep Learning Image Classification

This file contains the main FastAPI application that serves as the backend
for our image classification web app. It handles:
- Image uploads from the frontend
- Processing images with a deep learning model
- Returning predictions to the frontend

Key Concepts:
- FastAPI: Modern Python web framework for building APIs
- CORS: Allows frontend (different domain) to call this API
- Async: Non-blocking operations for better performance
"""

# Import required libraries
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import io
import logging

# Import our custom model class
from model import ImageClassifier

# Configure logging to help with debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# APPLICATION SETUP
# ============================================================================

# Create the FastAPI application instance
# This is the main object that handles all HTTP requests
app = FastAPI(
    title="Deep Learning Image Classification API",
    description="API for classifying images using PyTorch",
    version="1.0.0"
)

# ============================================================================
# CORS CONFIGURATION
# ============================================================================

# CORS (Cross-Origin Resource Sharing) allows our React frontend
# (running on a different domain/port) to make requests to this API
# Without CORS, browsers block these requests for security

app.add_middleware(
    CORSMiddleware,
    # allow_origins=["*"] means allow requests from ANY domain
    # In production, you should specify exact domains like:
    # allow_origins=["https://your-frontend.onrender.com"]
    allow_origins=["*"],

    # Allow browsers to send cookies/authentication
    allow_credentials=True,

    # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_methods=["*"],

    # Allow all headers in requests
    allow_headers=["*"],
)

# ============================================================================
# INITIALIZE THE MODEL
# ============================================================================

# Create an instance of our image classifier
# This loads the pre-trained PyTorch model into memory
# The model stays loaded for the lifetime of the application
try:
    logger.info("Loading the deep learning model...")
    classifier = ImageClassifier()
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    # If model fails to load, the app will still start but predictions will fail

# ============================================================================
# API ENDPOINTS (ROUTES)
# ============================================================================

@app.get("/")
async def root():
    """
    Root endpoint - Health check

    Purpose: Verify the API is running
    Method: GET
    URL: http://localhost:8000/

    Returns:
        JSON with welcome message and API status

    Example:
        curl http://localhost:8000/
    """
    return {
        "message": "Welcome to the Image Classification API!",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint

    Purpose: Check if the API and model are working
    Method: GET
    URL: http://localhost:8000/health

    Returns:
        JSON with health status

    Example:
        curl http://localhost:8000/health
    """
    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "message": "API is running and ready to accept requests"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Image classification endpoint

    Purpose: Receive an image and return classification predictions
    Method: POST
    URL: http://localhost:8000/predict

    Parameters:
        file: The uploaded image file (UploadFile object)
              Supported formats: JPEG, PNG, etc.

    Returns:
        JSON containing:
        - predictions: List of top predictions with labels and confidence scores
        - filename: Original filename
        - success: Boolean indicating if prediction was successful

    Example using curl:
        curl -X POST "http://localhost:8000/predict" \
             -H "accept: application/json" \
             -F "file=@cat.jpg"

    Example using Python requests:
        import requests
        files = {'file': open('cat.jpg', 'rb')}
        response = requests.post('http://localhost:8000/predict', files=files)
        print(response.json())
    """

    try:
        # Log the incoming request
        logger.info(f"Received prediction request for file: {file.filename}")

        # ----------------------------------------------------------------
        # STEP 1: Validate the uploaded file
        # ----------------------------------------------------------------

        # Check if a file was actually uploaded
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")

        # Check file type - only accept image files
        # file.content_type might be: 'image/jpeg', 'image/png', etc.
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"File must be an image. Received: {file.content_type}"
            )

        # ----------------------------------------------------------------
        # STEP 2: Read and process the image
        # ----------------------------------------------------------------

        # Read the file contents into memory
        # await is used because file reading is an async operation
        contents = await file.read()

        # Convert the bytes into a PIL Image object
        # PIL (Python Imaging Library) is used for image manipulation
        # io.BytesIO converts bytes into a file-like object
        image = Image.open(io.BytesIO(contents))

        # Convert image to RGB mode if it's not already
        # Some images might be in RGBA (with alpha channel) or grayscale
        # Our model expects RGB (3 channels: Red, Green, Blue)
        if image.mode != 'RGB':
            logger.info(f"Converting image from {image.mode} to RGB")
            image = image.convert('RGB')

        logger.info(f"Image loaded successfully: {image.size}")

        # ----------------------------------------------------------------
        # STEP 3: Make prediction using the model
        # ----------------------------------------------------------------

        # Call the predict method of our classifier
        # This will:
        # 1. Preprocess the image (resize, normalize)
        # 2. Convert to PyTorch tensor
        # 3. Run through the neural network
        # 4. Return top predictions
        predictions = classifier.predict(image)

        logger.info(f"Prediction successful: {predictions[0]['label']}")

        # ----------------------------------------------------------------
        # STEP 4: Return the results
        # ----------------------------------------------------------------

        # Return a JSON response with the predictions
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "predictions": predictions,
            "message": "Image classified successfully"
        })

    except HTTPException as he:
        # Re-raise HTTP exceptions (validation errors)
        raise he

    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Prediction error: {str(e)}")

        # Return a 500 Internal Server Error with details
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    finally:
        # Always close the uploaded file to free resources
        await file.close()


# ============================================================================
# TESTING ENDPOINT (Optional - useful for development)
# ============================================================================

@app.get("/model-info")
async def model_info():
    """
    Get information about the loaded model

    Purpose: Debug endpoint to check model details
    Method: GET
    URL: http://localhost:8000/model-info

    Returns:
        JSON with model information
    """
    return {
        "model_name": "ResNet-18",
        "framework": "PyTorch",
        "num_classes": 1000,
        "input_size": "224x224",
        "pretrained": True,
        "description": "Pre-trained on ImageNet dataset"
    }


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    This block runs when you execute: python app.py

    It starts the uvicorn server which runs the FastAPI application

    Parameters:
    - app: The FastAPI application object
    - host="0.0.0.0": Listen on all network interfaces
                      (allows external access, needed for deployment)
    - port=8000: Run on port 8000
    - reload=True: Auto-reload when code changes (development only)
    """

    # Print startup message
    print("=" * 70)
    print("Starting Image Classification API")
    print("=" * 70)
    print("API will be available at: http://localhost:8000")
    print("Interactive docs at: http://localhost:8000/docs")
    print("=" * 70)

    # Start the server
    uvicorn.run(
        "app:app",  # Format: "filename:app_variable"
        host="0.0.0.0",
        port=8000,
        reload=True  # Set to False in production
    )
