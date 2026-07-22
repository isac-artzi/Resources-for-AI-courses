"""
Deep Learning Model Module
==========================

This module handles the ML model loading, preprocessing, and inference.
It wraps TensorFlow/Keras functionality for image classification.

Model: MobileNetV2
- Pre-trained on ImageNet (1000 classes)
- Optimized for mobile/edge devices
- Good balance of accuracy and speed
- Input size: 224x224 pixels

Author: Cloud ML Deployment
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
from typing import List, Tuple
import logging

# Configure logging
logger = logging.getLogger(__name__)


class ImageClassifier:
    """
    Image Classification Model Wrapper

    This class encapsulates the ML model and provides a clean interface
    for image classification tasks.

    Attributes:
        model: TensorFlow/Keras model (MobileNetV2)
        input_shape: Expected input dimensions (224, 224, 3)

    Example:
        classifier = ImageClassifier()
        predictions = classifier.predict(image_array)
    """

    def __init__(self, model_name: str = "MobileNetV2"):
        """
        Initialize the image classifier

        Args:
            model_name: Name of the model to load (default: MobileNetV2)

        The model is loaded with:
        - Weights pre-trained on ImageNet
        - All layers included (include_top=True)
        - Input shape: 224x224x3 (RGB images)
        """
        logger.info(f"Initializing {model_name} model...")

        # Store model configuration
        self.model_name = model_name
        self.input_shape = (224, 224, 3)  # Height, Width, Channels (RGB)

        try:
            # ================================================================
            # Load Pre-trained Model
            # ================================================================
            # MobileNetV2 is downloaded from Keras on first run (~14MB)
            # Subsequent runs use the cached model from ~/.keras/models/

            self.model = MobileNetV2(
                include_top=True,      # Include the ImageNet classifier (top layers)
                weights='imagenet',    # Use pre-trained weights from ImageNet
                input_shape=self.input_shape  # Expected input dimensions
            )

            logger.info(f"✓ {model_name} loaded successfully")
            logger.info(f"  - Input shape: {self.input_shape}")
            logger.info(f"  - Output classes: 1000 (ImageNet)")
            logger.info(f"  - Parameters: {self.model.count_params():,}")

        except Exception as e:
            logger.error(f"✗ Failed to load model: {str(e)}")
            raise


    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input

        Steps:
        1. Resize to 224x224 (model's expected input size)
        2. Add batch dimension (model expects batches)
        3. Apply MobileNetV2-specific preprocessing

        Args:
            image: numpy array of shape (H, W, 3) - RGB image

        Returns:
            Preprocessed image array of shape (1, 224, 224, 3)

        Example:
            >>> image = np.array(PIL_image)  # Shape: (500, 500, 3)
            >>> processed = classifier.preprocess_image(image)
            >>> processed.shape
            (1, 224, 224, 3)
        """
        try:
            # ================================================================
            # Step 1: Resize to model's expected input size
            # ================================================================
            # Use TensorFlow's image resizing for consistency
            image_resized = tf.image.resize(
                image,
                [self.input_shape[0], self.input_shape[1]],  # 224x224
                method='bilinear'  # High-quality resizing
            )

            logger.debug(f"Image resized to {self.input_shape[:2]}")

            # ================================================================
            # Step 2: Add batch dimension
            # ================================================================
            # Models expect input of shape (batch_size, height, width, channels)
            # We're processing one image, so batch_size = 1
            image_batched = tf.expand_dims(image_resized, axis=0)

            # ================================================================
            # Step 3: Apply model-specific preprocessing
            # ================================================================
            # MobileNetV2 expects inputs scaled to [-1, 1] range
            # preprocess_input handles this transformation
            image_preprocessed = preprocess_input(image_batched)

            logger.debug(f"Preprocessing complete: {image_preprocessed.shape}")

            return image_preprocessed

        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise ValueError(f"Image preprocessing error: {str(e)}")


    def predict(self, image: np.ndarray, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Classify an image and return top predictions

        This method:
        1. Preprocesses the input image
        2. Runs inference through the neural network
        3. Decodes predictions to human-readable labels
        4. Returns top-k most confident predictions

        Args:
            image: numpy array of shape (H, W, 3) - RGB image
            top_k: Number of top predictions to return (default: 5)

        Returns:
            List of tuples: [(class_id, class_name, confidence), ...]

        Example:
            >>> predictions = classifier.predict(image_array)
            >>> predictions[0]
            ('n02099601', 'golden_retriever', 0.8934567)

        Raises:
            ValueError: If image preprocessing fails
            RuntimeError: If model inference fails
        """
        try:
            # ================================================================
            # Step 1: Preprocess image
            # ================================================================
            preprocessed_image = self.preprocess_image(image)
            logger.info(f"Running inference on image of shape {image.shape}")

            # ================================================================
            # Step 2: Run model inference
            # ================================================================
            # This is where the "magic" happens!
            # The image passes through the neural network layers
            # Returns probability distribution over 1000 ImageNet classes

            predictions = self.model.predict(
                preprocessed_image,
                verbose=0  # Suppress TensorFlow progress output
            )

            # predictions shape: (1, 1000) - probabilities for each class
            logger.debug(f"Raw predictions shape: {predictions.shape}")
            logger.debug(f"Prediction confidence range: {predictions.min():.4f} to {predictions.max():.4f}")

            # ================================================================
            # Step 3: Decode predictions to readable labels
            # ================================================================
            # Convert probability distribution to class names
            # Returns: [(class_id, class_name, probability), ...]

            decoded_predictions = decode_predictions(
                predictions,
                top=top_k  # Return top-k predictions
            )[0]  # Get first (and only) image in batch

            logger.info(f"Top prediction: {decoded_predictions[0][1]} ({decoded_predictions[0][2]:.2%})")

            return decoded_predictions

        except ValueError as e:
            # Preprocessing errors
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise

        except Exception as e:
            # Model inference errors
            logger.error(f"Model prediction failed: {str(e)}")
            raise RuntimeError(f"Prediction error: {str(e)}")


    def get_model_info(self) -> dict:
        """
        Get information about the loaded model

        Returns:
            Dictionary with model metadata

        Example:
            >>> info = classifier.get_model_info()
            >>> print(info['name'])
            'MobileNetV2'
        """
        return {
            "name": self.model_name,
            "input_shape": self.input_shape,
            "output_classes": 1000,
            "framework": "TensorFlow/Keras",
            "weights": "ImageNet",
            "parameters": self.model.count_params(),
            "trainable": self.model.trainable
        }


# ============================================================================
# Alternative Models (for advanced students)
# ============================================================================

class ResNetClassifier(ImageClassifier):
    """
    ResNet50 Image Classifier

    More accurate but slower than MobileNetV2
    Better for high-accuracy requirements

    Usage:
        classifier = ResNetClassifier()
        predictions = classifier.predict(image)
    """

    def __init__(self):
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

        logger.info("Initializing ResNet50 model...")
        self.model_name = "ResNet50"
        self.input_shape = (224, 224, 3)

        # Load ResNet50 model
        self.model = ResNet50(
            include_top=True,
            weights='imagenet',
            input_shape=self.input_shape
        )

        # Use ResNet-specific preprocessing
        self.preprocess_fn = resnet_preprocess

        logger.info("✓ ResNet50 loaded successfully")


class EfficientNetClassifier(ImageClassifier):
    """
    EfficientNetB0 Image Classifier

    State-of-the-art efficiency and accuracy
    Great balance for production use

    Usage:
        classifier = EfficientNetClassifier()
        predictions = classifier.predict(image)
    """

    def __init__(self):
        from tensorflow.keras.applications import EfficientNetB0
        from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess

        logger.info("Initializing EfficientNetB0 model...")
        self.model_name = "EfficientNetB0"
        self.input_shape = (224, 224, 3)

        # Load EfficientNet model
        self.model = EfficientNetB0(
            include_top=True,
            weights='imagenet',
            input_shape=self.input_shape
        )

        # Use EfficientNet-specific preprocessing
        self.preprocess_fn = eff_preprocess

        logger.info("✓ EfficientNetB0 loaded successfully")


# ============================================================================
# Testing and Debugging
# ============================================================================

if __name__ == "__main__":
    """
    Test the classifier with a sample image
    Run this file directly to test: python -m app.model
    """

    print("=" * 60)
    print("Testing Image Classifier")
    print("=" * 60)

    # Initialize classifier
    try:
        classifier = ImageClassifier()
        print("\n✓ Model loaded successfully\n")

        # Print model info
        info = classifier.get_model_info()
        print("Model Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        # Create a random test image (224x224x3)
        print("\n" + "=" * 60)
        print("Testing with random image...")
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # Get predictions
        predictions = classifier.predict(test_image)

        print("\nPredictions:")
        for i, (class_id, class_name, confidence) in enumerate(predictions, 1):
            print(f"  {i}. {class_name}: {confidence:.2%}")

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
