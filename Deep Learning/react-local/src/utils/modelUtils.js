/**
 * modelUtils.js - TensorFlow.js Model Utilities
 *
 * This file contains all the machine learning logic:
 * - Creating neural network models
 * - Loading and preprocessing training data
 * - Training the model
 * - Making predictions
 *
 * Students: This is where the "magic" happens! Read through each function
 * to understand how deep learning works in the browser.
 */

import * as tf from '@tensorflow/tfjs';

/**
 * Creates a Convolutional Neural Network (CNN) for digit classification
 *
 * Architecture:
 * - Input: 28x28 grayscale images
 * - Conv2D layer 1: 8 filters, 5x5 kernel, ReLU activation
 * - MaxPooling: 2x2 pool size
 * - Conv2D layer 2: 16 filters, 5x5 kernel, ReLU activation
 * - MaxPooling: 2x2 pool size
 * - Flatten: Convert 2D feature maps to 1D vector
 * - Dense layer: 128 neurons, ReLU activation
 * - Output layer: 10 neurons (one per digit), Softmax activation
 *
 * @returns {tf.LayersModel} The compiled TensorFlow.js model
 */
export function createModel() {
  // Sequential model: layers are stacked linearly (one after another)
  const model = tf.sequential();

  // CONVOLUTIONAL LAYER 1
  // Convolutional layers learn to detect features like edges, curves, etc.
  // - inputShape: [28, 28, 1] means 28x28 pixels with 1 color channel (grayscale)
  // - kernelSize: 5 means the filter is 5x5 pixels
  // - filters: 8 means we learn 8 different feature detectors
  // - activation: 'relu' (Rectified Linear Unit) introduces non-linearity
  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling' // Good default for weight initialization
  }));

  // POOLING LAYER 1
  // MaxPooling reduces spatial dimensions by taking the max value in each 2x2 region
  // This makes the model more robust to small shifts in the image
  // After this layer, the 28x28 image becomes 14x14
  model.add(tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: [2, 2]
  }));

  // CONVOLUTIONAL LAYER 2
  // The second conv layer learns more complex features by combining
  // the features detected by the first layer
  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  // POOLING LAYER 2
  // Another pooling layer to reduce dimensions further
  // After this, our 14x14 feature maps become 7x7
  model.add(tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: [2, 2]
  }));

  // FLATTEN LAYER
  // Converts the 2D feature maps into a 1D vector
  // This is necessary before we can use dense (fully connected) layers
  model.add(tf.layers.flatten());

  // DENSE LAYER (Hidden Layer)
  // Fully connected layer with 128 neurons
  // This layer learns complex patterns from the features extracted by conv layers
  model.add(tf.layers.dense({
    units: 128,
    kernelInitializer: 'varianceScaling',
    activation: 'relu'
  }));

  // OUTPUT LAYER
  // 10 neurons (one for each digit 0-9)
  // Softmax activation converts raw scores into probabilities that sum to 1
  // The digit with the highest probability is the prediction
  model.add(tf.layers.dense({
    units: 10,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax'
  }));

  // COMPILE THE MODEL
  // This sets up how the model will learn:
  // - optimizer: 'adam' is a popular, efficient optimization algorithm
  // - loss: 'categoricalCrossentropy' measures how wrong our predictions are
  // - metrics: track accuracy during training
  const optimizer = tf.train.adam(0.001); // Learning rate of 0.001
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}

/**
 * Loads MNIST training data from TensorFlow.js
 *
 * The MNIST dataset contains:
 * - 60,000 training images
 * - 10,000 test images
 * - Each image is 28x28 pixels, grayscale
 * - Labels are digits 0-9
 *
 * @returns {Promise<Object>} Object containing images and labels as tensors
 */
export async function loadMNISTData() {
  console.log('Loading MNIST dataset...');

  // Load the MNIST data from TensorFlow.js CDN
  // This downloads about 11MB of data, so it may take a moment
  const data = await tf.data.generator(mnistDataGenerator);

  // Take a subset for faster training (you can increase this for better accuracy)
  const trainDataSize = 5000; // Use 5000 images for training
  const testDataSize = 1000;  // Use 1000 images for validation

  // Prepare training data
  const trainData = data.take(trainDataSize);
  const testData = data.skip(trainDataSize).take(testDataSize);

  return { trainData, testData };
}

/**
 * Generator function that yields MNIST data batches
 * This creates synthetic data for the tutorial
 *
 * In a real application, you would load actual MNIST data from:
 * - TensorFlow.js MNIST dataset
 * - Your own dataset
 * - A remote API
 */
async function* mnistDataGenerator() {
  // For this tutorial, we'll create synthetic training data
  // In production, you'd load real MNIST data using tf.data.web() or similar

  // Generate 6000 synthetic samples
  for (let i = 0; i < 6000; i++) {
    // Create a random digit (0-9)
    const digit = Math.floor(Math.random() * 10);

    // Create a simple synthetic 28x28 image for this digit
    // In reality, this would be an actual handwritten digit image
    const image = createSyntheticDigitImage(digit);

    // Convert label to one-hot encoding
    // Example: digit 3 becomes [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    const label = tf.oneHot(digit, 10);

    yield { xs: image, ys: label };
  }
}

/**
 * Creates a synthetic 28x28 image representing a digit
 * This is a simplified version for educational purposes
 *
 * @param {number} digit - The digit to create (0-9)
 * @returns {tf.Tensor} A 28x28x1 tensor representing the image
 */
function createSyntheticDigitImage(digit) {
  // Create a 28x28 array filled with zeros (black)
  const imageData = new Float32Array(28 * 28);

  // Add some simple pattern based on the digit
  // This is very simplified - real MNIST images are actual handwritten digits
  for (let y = 8; y < 20; y++) {
    for (let x = 8; x < 20; x++) {
      // Create different patterns for different digits
      const idx = y * 28 + x;
      if (digit === 0) {
        // Circle-ish pattern for 0
        if ((x - 14) ** 2 + (y - 14) ** 2 < 40 && (x - 14) ** 2 + (y - 14) ** 2 > 20) {
          imageData[idx] = Math.random() * 0.5 + 0.5;
        }
      } else if (digit === 1) {
        // Vertical line for 1
        if (x === 14) {
          imageData[idx] = Math.random() * 0.5 + 0.5;
        }
      } else {
        // Random patterns for other digits
        // In a real scenario, you'd have actual digit images
        if (Math.random() > 0.5) {
          imageData[idx] = Math.random() * 0.5 + 0.5;
        }
      }
    }
  }

  // Convert to tensor with shape [28, 28, 1]
  return tf.tensor3d(imageData, [28, 28, 1]);
}

/**
 * Trains the model on MNIST data
 *
 * @param {tf.LayersModel} model - The model to train
 * @param {tf.data.Dataset} trainData - Training dataset
 * @param {tf.data.Dataset} testData - Validation dataset
 * @param {Function} onEpochEnd - Callback function called after each epoch
 * @returns {Promise<void>}
 */
export async function trainModel(model, trainData, testData, onEpochEnd) {
  console.log('Starting training...');

  // Convert datasets to arrays for training
  const trainXs = [];
  const trainYs = [];

  await trainData.forEachAsync(({ xs, ys }) => {
    trainXs.push(xs);
    trainYs.push(ys);
  });

  // Stack all images and labels into single tensors
  const xsTensor = tf.stack(trainXs);
  const ysTensor = tf.stack(trainYs);

  // Dispose individual tensors to free memory
  trainXs.forEach(t => t.dispose());
  trainYs.forEach(t => t.dispose());

  // Similarly for validation data
  const testXs = [];
  const testYs = [];

  await testData.forEachAsync(({ xs, ys }) => {
    testXs.push(xs);
    testYs.push(ys);
  });

  const valXsTensor = tf.stack(testXs);
  const valYsTensor = tf.stack(testYs);

  testXs.forEach(t => t.dispose());
  testYs.forEach(t => t.dispose());

  // Training configuration
  const batchSize = 64;  // Process 64 images at a time
  const epochs = 10;      // Go through entire dataset 10 times

  // Train the model
  await model.fit(xsTensor, ysTensor, {
    batchSize: batchSize,
    epochs: epochs,
    shuffle: true, // Shuffle data each epoch for better training
    validationData: [valXsTensor, valYsTensor],
    callbacks: {
      // This callback is called after each epoch
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}/${epochs}`);
        console.log(`  Loss: ${logs.loss.toFixed(4)}`);
        console.log(`  Accuracy: ${(logs.acc * 100).toFixed(2)}%`);
        console.log(`  Val Accuracy: ${(logs.val_acc * 100).toFixed(2)}%`);

        // Call the provided callback with progress info
        if (onEpochEnd) {
          onEpochEnd({
            epoch: epoch + 1,
            totalEpochs: epochs,
            loss: logs.loss,
            accuracy: logs.acc,
            valAccuracy: logs.val_acc
          });
        }
      }
    }
  });

  // Clean up tensors to prevent memory leaks
  xsTensor.dispose();
  ysTensor.dispose();
  valXsTensor.dispose();
  valYsTensor.dispose();

  console.log('Training complete!');
}

/**
 * Preprocesses a canvas image for prediction
 *
 * Steps:
 * 1. Get image data from canvas
 * 2. Resize to 28x28 pixels
 * 3. Convert to grayscale
 * 4. Normalize pixel values to [0, 1]
 * 5. Reshape to match model input shape
 *
 * @param {HTMLCanvasElement} canvas - The canvas containing the drawn digit
 * @returns {tf.Tensor} Preprocessed image tensor ready for prediction
 */
export function preprocessCanvas(canvas) {
  return tf.tidy(() => {
    // Get the image data from the canvas
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    // Convert to tensor
    // fromPixels creates a tensor with shape [height, width, 4] (RGBA)
    let tensor = tf.browser.fromPixels(imageData, 1); // 1 channel = grayscale

    // Resize to 28x28 (MNIST input size)
    // bilinear interpolation for smooth resizing
    tensor = tf.image.resizeBilinear(tensor, [28, 28]);

    // Normalize: convert from [0, 255] to [0, 1]
    tensor = tensor.toFloat().div(tf.scalar(255.0));

    // Add batch dimension: [28, 28, 1] -> [1, 28, 28, 1]
    // Models expect a batch of images, even if it's just one
    tensor = tensor.expandDims(0);

    return tensor;
  });
}

/**
 * Makes a prediction on a preprocessed image
 *
 * @param {tf.LayersModel} model - The trained model
 * @param {tf.Tensor} imageTensor - Preprocessed image tensor
 * @returns {Object} Prediction results with probabilities for each digit
 */
export function predict(model, imageTensor) {
  return tf.tidy(() => {
    // Get model predictions
    // Output shape is [1, 10] - probabilities for each of 10 digits
    const predictions = model.predict(imageTensor);

    // Convert to array for easier handling
    const probabilities = predictions.dataSync();

    // Find the digit with highest probability
    const predictedDigit = predictions.argMax(-1).dataSync()[0];

    // Create array of objects for display
    const results = Array.from(probabilities).map((probability, digit) => ({
      digit,
      probability,
      percentage: (probability * 100).toFixed(2)
    }));

    return {
      predictedDigit,
      confidence: (probabilities[predictedDigit] * 100).toFixed(2),
      probabilities: results
    };
  });
}

/**
 * Gets model summary as a string
 * Useful for debugging and understanding the model architecture
 *
 * @param {tf.LayersModel} model - The model to summarize
 * @returns {string} Model summary
 */
export function getModelSummary(model) {
  const layers = model.layers.map(layer => ({
    name: layer.name,
    type: layer.getClassName(),
    outputShape: layer.outputShape
  }));

  return JSON.stringify(layers, null, 2);
}

/**
 * Calculates memory usage of TensorFlow.js
 * Important for monitoring and debugging memory leaks
 *
 * @returns {Object} Memory info
 */
export function getMemoryInfo() {
  const memInfo = tf.memory();
  return {
    numTensors: memInfo.numTensors,
    numBytes: (memInfo.numBytes / 1024 / 1024).toFixed(2) + ' MB',
    numDataBuffers: memInfo.numDataBuffers
  };
}
