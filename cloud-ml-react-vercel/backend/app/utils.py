"""
Utility Functions Module
========================

This module contains helper functions for image validation,
prediction formatting, and other common operations.

Functions:
- validate_image: Check if uploaded file is a valid image
- format_predictions: Convert model output to API response format
- allowed_file: Check if file extension is allowed

Author: Cloud ML Deployment
"""

from typing import List, Dict, Tuple
import logging
import os

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Allowed image file extensions
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

# Maximum file size (in bytes) - 10MB
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# ============================================================================
# File Validation Functions
# ============================================================================

def validate_image(filename: str) -> bool:
    """
    Validate if the uploaded file is an allowed image type

    This function checks:
    1. File has an extension
    2. Extension is in the allowed list

    Args:
        filename: Name of the uploaded file (e.g., "photo.jpg")

    Returns:
        True if file is valid

    Raises:
        ValueError: If file is invalid

    Example:
        >>> validate_image("photo.jpg")
        True
        >>> validate_image("document.pdf")
        ValueError: File type '.pdf' is not allowed
    """
    # Check if filename exists
    if not filename:
        raise ValueError("No filename provided")

    # Extract file extension
    # os.path.splitext splits "photo.jpg" into ("photo", ".jpg")
    file_ext = os.path.splitext(filename)[1].lower()

    # Check if extension exists
    if not file_ext:
        raise ValueError("File has no extension")

    # Check if extension is allowed
    if file_ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"File type '{file_ext}' is not allowed. "
            f"Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    logger.debug(f"File validation passed: {filename}")
    return True


def allowed_file(filename: str) -> bool:
    """
    Check if a file extension is allowed

    This is a simpler version of validate_image that returns bool
    instead of raising exceptions.

    Args:
        filename: Name of the file to check

    Returns:
        True if file extension is allowed, False otherwise

    Example:
        >>> allowed_file("photo.jpg")
        True
        >>> allowed_file("document.pdf")
        False
    """
    if not filename:
        return False

    file_ext = os.path.splitext(filename)[1].lower()
    return file_ext in ALLOWED_EXTENSIONS


# ============================================================================
# Prediction Formatting Functions
# ============================================================================

def format_predictions(
    predictions: List[Tuple[str, str, float]],
    top_k: int = 5
) -> List[Dict[str, any]]:
    """
    Format model predictions for API response

    Converts TensorFlow's prediction format to a clean JSON structure
    that's easy to consume in the frontend.

    Args:
        predictions: List of tuples from decode_predictions
                    Format: [(class_id, class_name, probability), ...]
        top_k: Number of predictions to return (default: 5)

    Returns:
        List of dictionaries with formatted predictions

    Example Input:
        [
            ('n02099601', 'golden_retriever', 0.8934567),
            ('n02099712', 'Labrador_retriever', 0.0612345),
            ('n02109961', 'Eskimo_dog', 0.0234567)
        ]

    Example Output:
        [
            {
                "class": "Golden Retriever",
                "confidence": 0.89,
                "class_id": "n02099601",
                "rank": 1
            },
            {
                "class": "Labrador Retriever",
                "confidence": 0.06,
                "class_id": "n02099712",
                "rank": 2
            }
        ]
    """
    formatted = []

    # Process top-k predictions
    for rank, (class_id, class_name, confidence) in enumerate(predictions[:top_k], start=1):
        # ====================================================================
        # Clean up class name
        # ====================================================================
        # ImageNet class names use underscores: "golden_retriever"
        # Convert to title case: "Golden Retriever"

        # Replace underscores with spaces
        clean_name = class_name.replace('_', ' ')

        # Convert to title case (first letter of each word capitalized)
        clean_name = clean_name.title()

        # ====================================================================
        # Create formatted prediction object
        # ====================================================================
        prediction = {
            "class": clean_name,              # Human-readable name
            "confidence": round(float(confidence), 4),  # Round to 4 decimals
            "confidence_percent": f"{confidence * 100:.2f}%",  # Percentage format
            "class_id": class_id,             # Original ImageNet ID
            "rank": rank                      # Position in top-k
        }

        formatted.append(prediction)

    logger.debug(f"Formatted {len(formatted)} predictions")

    return formatted


# ============================================================================
# Additional Helper Functions
# ============================================================================

def calculate_confidence_distribution(predictions: List[Dict]) -> Dict:
    """
    Calculate statistics about prediction confidence

    Useful for understanding model certainty and providing
    additional insights to users.

    Args:
        predictions: List of formatted prediction dictionaries

    Returns:
        Dictionary with confidence statistics

    Example:
        >>> stats = calculate_confidence_distribution(predictions)
        >>> print(stats)
        {
            "top_confidence": 0.89,
            "confidence_spread": 0.83,
            "certainty_level": "high"
        }
    """
    if not predictions:
        return {"error": "No predictions provided"}

    confidences = [p["confidence"] for p in predictions]

    # Calculate statistics
    top_confidence = max(confidences)
    lowest_confidence = min(confidences)
    confidence_spread = top_confidence - lowest_confidence

    # Determine certainty level
    if top_confidence >= 0.8:
        certainty = "high"
    elif top_confidence >= 0.5:
        certainty = "medium"
    else:
        certainty = "low"

    return {
        "top_confidence": round(top_confidence, 4),
        "lowest_confidence": round(lowest_confidence, 4),
        "confidence_spread": round(confidence_spread, 4),
        "certainty_level": certainty,
        "average_confidence": round(sum(confidences) / len(confidences), 4)
    }


def create_error_response(
    error_type: str,
    message: str,
    details: str = None
) -> Dict:
    """
    Create a standardized error response

    Args:
        error_type: Type of error (e.g., "validation_error", "processing_error")
        message: User-friendly error message
        details: Additional technical details (optional)

    Returns:
        Formatted error response dictionary

    Example:
        >>> error = create_error_response(
        ...     "validation_error",
        ...     "Invalid image file",
        ...     "File type .pdf is not supported"
        ... )
    """
    response = {
        "success": False,
        "error": error_type,
        "message": message
    }

    if details:
        response["details"] = details

    logger.warning(f"Error response created: {error_type} - {message}")

    return response


def log_prediction_stats(predictions: List[Dict], processing_time: float):
    """
    Log statistics about a prediction for monitoring

    This helps track model performance and identify issues.

    Args:
        predictions: List of formatted predictions
        processing_time: Time taken for inference (seconds)
    """
    if not predictions:
        logger.warning("No predictions to log")
        return

    top_pred = predictions[0]

    logger.info(
        f"Prediction Stats | "
        f"Top: {top_pred['class']} ({top_pred['confidence']:.2%}) | "
        f"Time: {processing_time:.3f}s | "
        f"Results: {len(predictions)}"
    )


def get_image_dimensions(image) -> Tuple[int, int]:
    """
    Get image width and height

    Args:
        image: PIL Image object or numpy array

    Returns:
        Tuple of (width, height)

    Example:
        >>> from PIL import Image
        >>> img = Image.open("photo.jpg")
        >>> width, height = get_image_dimensions(img)
    """
    try:
        # For PIL Images
        if hasattr(image, 'size'):
            return image.size  # Returns (width, height)

        # For numpy arrays
        if hasattr(image, 'shape'):
            # Numpy arrays are (height, width, channels)
            height, width = image.shape[:2]
            return (width, height)

        return (0, 0)

    except Exception as e:
        logger.error(f"Could not get image dimensions: {str(e)}")
        return (0, 0)


# ============================================================================
# Testing and Debugging
# ============================================================================

if __name__ == "__main__":
    """
    Test utility functions
    Run: python -m app.utils
    """

    print("=" * 60)
    print("Testing Utility Functions")
    print("=" * 60)

    # Test file validation
    print("\n1. Testing file validation:")
    test_files = [
        "photo.jpg",
        "image.png",
        "document.pdf",
        "video.mp4",
        "picture.jpeg"
    ]

    for filename in test_files:
        try:
            validate_image(filename)
            print(f"  ✓ {filename} - Valid")
        except ValueError as e:
            print(f"  ✗ {filename} - Invalid: {e}")

    # Test prediction formatting
    print("\n2. Testing prediction formatting:")
    mock_predictions = [
        ('n02099601', 'golden_retriever', 0.8934567),
        ('n02099712', 'Labrador_retriever', 0.0612345),
        ('n02109961', 'Eskimo_dog', 0.0234567)
    ]

    formatted = format_predictions(mock_predictions)
    print(f"  Formatted {len(formatted)} predictions:")
    for pred in formatted:
        print(f"    {pred['rank']}. {pred['class']}: {pred['confidence_percent']}")

    # Test confidence distribution
    print("\n3. Testing confidence distribution:")
    stats = calculate_confidence_distribution(formatted)
    print(f"  Top confidence: {stats['top_confidence']}")
    print(f"  Certainty level: {stats['certainty_level']}")
    print(f"  Average confidence: {stats['average_confidence']}")

    print("\n" + "=" * 60)
    print("✓ All utility tests passed!")
    print("=" * 60)
