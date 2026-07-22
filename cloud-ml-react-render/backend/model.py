"""
Deep Learning Model for Image Classification

This file contains the ImageClassifier class that wraps a PyTorch model
and provides easy-to-use methods for making predictions.

Key Concepts:
- PyTorch: Popular deep learning framework
- Transfer Learning: Using a pre-trained model (ResNet-18)
- ImageNet: Dataset with 1000 categories the model was trained on
- Preprocessing: Converting images to the format the model expects
"""

import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json
import logging

# Configure logging
logger = logging.getLogger(__name__)


class ImageClassifier:
    """
    Image Classification using PyTorch

    This class encapsulates all the deep learning functionality:
    - Loading a pre-trained model
    - Preprocessing images
    - Making predictions
    - Formatting results

    Model: ResNet-18
    - ResNet: Residual Network, a type of convolutional neural network
    - 18: Number of layers (there are also ResNet-50, ResNet-101, etc.)
    - Pre-trained on ImageNet: 1.2 million images, 1000 categories

    Why ResNet-18?
    - Good accuracy for its size
    - Fast inference (predictions)
    - Widely used and well-tested
    - Perfect for learning
    """

    def __init__(self, model_name='resnet18'):
        """
        Initialize the classifier

        This method:
        1. Loads the pre-trained model
        2. Sets up image preprocessing
        3. Loads class labels
        4. Prepares the model for inference

        Args:
            model_name: Name of the model to use (default: 'resnet18')
        """
        logger.info(f"Initializing ImageClassifier with {model_name}")

        # ----------------------------------------------------------------
        # STEP 1: Set up the device (CPU or GPU)
        # ----------------------------------------------------------------

        # Check if CUDA (NVIDIA GPU) is available
        # GPU is much faster for deep learning, but CPU works fine too
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # ----------------------------------------------------------------
        # STEP 2: Load the pre-trained model
        # ----------------------------------------------------------------

        # Load ResNet-18 with pre-trained weights
        # pretrained=True downloads weights trained on ImageNet
        # This is called "transfer learning" - using knowledge from one task
        # (ImageNet classification) for another task

        logger.info("Loading pre-trained ResNet-18 model...")
        self.model = models.resnet18(pretrained=True)

        # Move model to the device (CPU or GPU)
        self.model = self.model.to(self.device)

        # Set model to evaluation mode
        # This disables layers like dropout and batch normalization
        # which behave differently during training vs. inference
        self.model.eval()

        logger.info("Model loaded successfully!")

        # ----------------------------------------------------------------
        # STEP 3: Define image preprocessing transforms
        # ----------------------------------------------------------------

        # Neural networks expect images in a specific format:
        # - Fixed size (224x224 for ResNet)
        # - Normalized pixel values (mean and std from ImageNet)
        # - Tensor format (not PIL Image)

        self.preprocess = transforms.Compose([
            # Resize image to 256 pixels on the shorter side
            # This is larger than needed to allow for center cropping
            transforms.Resize(256),

            # Crop the center 224x224 region
            # ResNet was trained on 224x224 images
            transforms.CenterCrop(224),

            # Convert PIL Image to PyTorch tensor
            # Also converts pixel values from [0, 255] to [0, 1]
            transforms.ToTensor(),

            # Normalize using ImageNet statistics
            # These values are the mean and std of ImageNet dataset
            # Normalization helps the model perform better
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # RGB means
                std=[0.229, 0.224, 0.225]     # RGB standard deviations
            ),
        ])

        # ----------------------------------------------------------------
        # STEP 4: Load ImageNet class labels
        # ----------------------------------------------------------------

        # The model outputs 1000 numbers (one for each ImageNet class)
        # We need labels to convert these numbers to readable names
        # For example: class 281 = "tabby cat"

        self.labels = self._load_imagenet_labels()
        logger.info(f"Loaded {len(self.labels)} class labels")


    def _load_imagenet_labels(self):
        """
        Load ImageNet class labels

        Returns a list of 1000 class names in the correct order

        ImageNet has 1000 categories including:
        - Animals (dogs, cats, birds, etc.)
        - Vehicles (cars, planes, etc.)
        - Objects (furniture, instruments, etc.)
        - Food items
        - And more...

        Returns:
            list: 1000 class labels
        """
        # This is a simplified list of ImageNet labels
        # In a production system, you'd load this from a JSON file
        # For now, we'll create a basic list

        # You can download the full list from:
        # https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt

        try:
            # Try to load from file if it exists
            with open('imagenet_classes.txt', 'r') as f:
                labels = [line.strip() for line in f.readlines()]
                return labels
        except FileNotFoundError:
            # If file doesn't exist, use a simplified version
            # These are just some example classes - the model knows all 1000
            logger.warning("imagenet_classes.txt not found, using simplified labels")
            return [
                "tench", "goldfish", "great white shark", "tiger shark",
                "hammerhead", "electric ray", "stingray", "cock", "hen",
                "ostrich", "brambling", "goldfinch", "house finch", "junco",
                "indigo bunting", "robin", "bulbul", "jay", "magpie",
                "chickadee", "water ouzel", "kite", "bald eagle", "vulture",
                "great grey owl", "European fire salamander", "common newt",
                "eft", "spotted salamander", "axolotl", "bullfrog", "tree frog",
                # ... (normally 1000 labels)
                # For simplicity, we'll let the model use class indices
            ] + [f"class_{i}" for i in range(33, 1000)]


    def predict(self, image, top_k=5):
        """
        Make a prediction on an image

        This is the main method that:
        1. Preprocesses the image
        2. Runs it through the neural network
        3. Returns the top predictions

        Args:
            image: PIL Image object
            top_k: Number of top predictions to return (default: 5)

        Returns:
            list: Top predictions with labels and confidence scores
                  Example: [
                      {"label": "tabby cat", "confidence": 0.92},
                      {"label": "Egyptian cat", "confidence": 0.05},
                      ...
                  ]

        How Neural Networks Make Predictions:
        1. Image goes through many layers (convolutions, pooling, etc.)
        2. Each layer extracts features (edges, shapes, patterns)
        3. Final layer outputs 1000 numbers (logits)
        4. Softmax converts logits to probabilities (sum to 1.0)
        5. Highest probabilities are the predicted classes
        """

        # ----------------------------------------------------------------
        # STEP 1: Preprocess the image
        # ----------------------------------------------------------------

        # Apply the preprocessing transforms we defined in __init__
        # This converts the PIL Image to a normalized tensor
        input_tensor = self.preprocess(image)

        # Add a batch dimension
        # Neural networks expect batches of images: [batch_size, channels, height, width]
        # Even for a single image, we need a batch dimension: [1, 3, 224, 224]
        input_batch = input_tensor.unsqueeze(0)

        # Move tensor to the same device as the model (CPU or GPU)
        input_batch = input_batch.to(self.device)

        # ----------------------------------------------------------------
        # STEP 2: Make prediction (forward pass)
        # ----------------------------------------------------------------

        # Disable gradient computation
        # Gradients are only needed for training, not inference
        # This saves memory and speeds up computation
        with torch.no_grad():
            # Run the image through the model
            # output shape: [1, 1000] (1 image, 1000 class scores)
            output = self.model(input_batch)

        # ----------------------------------------------------------------
        # STEP 3: Process the output
        # ----------------------------------------------------------------

        # Apply softmax to convert raw scores (logits) to probabilities
        # Softmax ensures all probabilities sum to 1.0
        # dim=1 means apply softmax across the class dimension
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get the top k predictions
        # torch.topk returns (values, indices)
        # values: the top k probability scores
        # indices: the corresponding class indices
        top_probs, top_indices = torch.topk(probabilities, top_k)

        # ----------------------------------------------------------------
        # STEP 4: Format results
        # ----------------------------------------------------------------

        # Convert to Python lists (from PyTorch tensors)
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        # Create a list of predictions with labels and confidence scores
        predictions = []
        for i in range(top_k):
            class_idx = int(top_indices[i])
            confidence = float(top_probs[i])

            # Get the class label
            # Make sure the index is within bounds
            if class_idx < len(self.labels):
                label = self.labels[class_idx]
            else:
                label = f"Class {class_idx}"

            predictions.append({
                "label": label,
                "confidence": round(confidence, 4),  # Round to 4 decimal places
                "class_id": class_idx
            })

        return predictions


    def predict_from_path(self, image_path, top_k=5):
        """
        Convenience method to predict from an image file path

        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return

        Returns:
            list: Top predictions with labels and confidence scores

        Example:
            classifier = ImageClassifier()
            predictions = classifier.predict_from_path('cat.jpg')
            print(predictions)
        """
        # Open the image file
        image = Image.open(image_path)

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Make prediction
        return self.predict(image, top_k)


# ============================================================================
# TESTING CODE (Optional - for standalone testing)
# ============================================================================

if __name__ == "__main__":
    """
    This block runs when you execute: python model.py

    It's useful for testing the model without running the full API
    """
    print("=" * 70)
    print("Testing Image Classifier")
    print("=" * 70)

    # Create classifier instance
    classifier = ImageClassifier()

    # Test with a sample image
    # You can replace this with any image path on your computer
    test_image_path = "test_image.jpg"

    try:
        print(f"\nTesting with image: {test_image_path}")
        predictions = classifier.predict_from_path(test_image_path, top_k=5)

        print("\nPredictions:")
        print("-" * 70)
        for i, pred in enumerate(predictions, 1):
            print(f"{i}. {pred['label']}: {pred['confidence']:.2%}")
        print("-" * 70)

    except FileNotFoundError:
        print(f"\nError: Could not find image at {test_image_path}")
        print("Please provide a valid image path to test the classifier")
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
