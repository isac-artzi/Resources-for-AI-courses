"""
Deep Learning Model Handler

This module handles loading and running inference with a TensorFlow model.
Uses MobileNetV2 pre-trained on ImageNet for image classification.

Author: Your Name
Date: 2026
"""

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class ModelHandler:
    """
    Handles deep learning model operations including loading and inference
    """

    def __init__(self):
        """Initialize the model handler"""
        self.model = None
        self.model_name = "MobileNetV2"
        self.input_shape = (224, 224)

    def load_model(self):
        """
        Load the pre-trained MobileNetV2 model

        This uses a pre-trained model from TensorFlow/Keras.
        For custom models, replace this with your own model loading logic.
        """
        try:
            logger.info("Loading MobileNetV2 model...")

            # Load pre-trained MobileNetV2
            # weights='imagenet' downloads the pre-trained weights
            self.model = MobileNetV2(
                weights='imagenet',
                include_top=True,
                input_shape=(224, 224, 3)
            )

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def preprocess_image(self, img: Image.Image) -> np.ndarray:
        """
        Preprocess image for model input

        Args:
            img (PIL.Image): Input image

        Returns:
            np.ndarray: Preprocessed image array ready for model
        """
        # Resize image to model input size
        img = img.resize(self.input_shape)

        # Convert to array
        img_array = image.img_to_array(img)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocess according to MobileNetV2 requirements
        img_array = preprocess_input(img_array)

        return img_array

    def predict(self, img: Image.Image, top_k: int = 5) -> List[Dict[str, float]]:
        """
        Make prediction on an image

        Args:
            img (PIL.Image): Input image
            top_k (int): Number of top predictions to return

        Returns:
            List[Dict]: List of predictions with class names and confidence scores

        Example:
            [
                {"class": "golden_retriever", "confidence": 0.95},
                {"class": "labrador", "confidence": 0.03}
            ]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Preprocess image
            processed_img = self.preprocess_image(img)

            # Make prediction
            predictions = self.model.predict(processed_img, verbose=0)

            # Decode predictions to human-readable labels
            decoded_predictions = decode_predictions(predictions, top=top_k)[0]

            # Format results
            results = []
            for pred in decoded_predictions:
                class_id, class_name, confidence = pred
                results.append({
                    "class": class_name,
                    "confidence": float(confidence),
                    "class_id": class_id
                })

            return results

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the loaded model

        Returns:
            dict: Model information
        """
        return {
            "model_name": self.model_name,
            "input_shape": self.input_shape,
            "loaded": self.model is not None,
            "framework": "TensorFlow",
            "version": tf.__version__
        }


# Alternative: Custom Model Handler
class CustomModelHandler:
    """
    Template for loading custom trained models

    Use this as a starting point for deploying your own models.
    """

    def __init__(self, model_path: str):
        """
        Initialize with path to custom model

        Args:
            model_path (str): Path to saved model (.h5 or SavedModel format)
        """
        self.model_path = model_path
        self.model = None

    def load_model(self):
        """Load custom model from file"""
        try:
            logger.info(f"Loading custom model from {self.model_path}")

            # Load saved model
            self.model = tf.keras.models.load_model(self.model_path)

            logger.info("Custom model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading custom model: {str(e)}")
            raise

    def predict(self, img: Image.Image) -> List[Dict[str, float]]:
        """
        Make prediction with custom model

        Implement your own preprocessing and prediction logic here
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Implement your preprocessing
        # processed_img = your_preprocessing_function(img)

        # Make prediction
        # predictions = self.model.predict(processed_img)

        # Format and return results
        # return formatted_results

        pass


# Example: PyTorch Model Handler
class PyTorchModelHandler:
    """
    Template for PyTorch models

    If you prefer PyTorch over TensorFlow, use this template.
    """

    def __init__(self, model_path: str = None):
        """Initialize PyTorch model handler"""
        self.model_path = model_path
        self.model = None
        self.device = None

    def load_model(self):
        """Load PyTorch model"""
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms

            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load pre-trained model (example: ResNet)
            self.model = models.resnet50(pretrained=True)
            self.model.to(self.device)
            self.model.eval()

            # Define transforms
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            logger.info("PyTorch model loaded successfully")

        except ImportError:
            logger.error("PyTorch not installed. Install with: pip install torch torchvision")
            raise
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {str(e)}")
            raise

    def predict(self, img: Image.Image) -> List[Dict[str, float]]:
        """Make prediction with PyTorch model"""
        import torch

        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Preprocess image
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        # Get top 5 predictions
        top5_prob, top5_catid = torch.topk(probabilities, 5)

        # Format results
        results = []
        for i in range(top5_prob.size(0)):
            results.append({
                "class_id": int(top5_catid[i]),
                "confidence": float(top5_prob[i])
            })

        return results
