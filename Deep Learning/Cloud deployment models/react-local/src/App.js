/**
 * App.js - Main Application Component
 *
 * This is the root component of our React deep learning application.
 * It coordinates all the child components and manages the application state.
 *
 * Component Structure:
 * - Header: Title and description
 * - ModelTrainer: Handles model creation and training
 * - DrawingCanvas: Interactive canvas for drawing digits
 * - PredictionDisplay: Shows prediction results
 *
 * React Concepts Used:
 * - useState: Managing application state (model, predictions)
 * - Component composition: Combining multiple components
 * - Props and callbacks: Parent-child communication
 * - Conditional rendering: Showing/hiding UI based on state
 */

import React, { useState } from 'react';
import './App.css';
import DrawingCanvas from './components/DrawingCanvas';
import ModelTrainer from './components/ModelTrainer';
import PredictionDisplay from './components/PredictionDisplay';
import { preprocessCanvas, predict } from './utils/modelUtils';

/**
 * Main App Component
 */
function App() {
  // STATE MANAGEMENT
  // ----------------

  // Store the trained TensorFlow.js model
  // Initially null, set after training completes
  const [model, setModel] = useState(null);

  // Store the current prediction results
  // Updated whenever user draws on the canvas
  const [prediction, setPrediction] = useState(null);

  // CALLBACK FUNCTIONS
  // ------------------

  /**
   * Called when the model finishes training
   * Saves the model to state so we can use it for predictions
   *
   * @param {tf.LayersModel} trainedModel - The trained model
   */
  const handleModelReady = (trainedModel) => {
    console.log('Model is ready for predictions!');
    setModel(trainedModel);
  };

  /**
   * Called whenever the canvas changes (user draws or clears)
   * Makes a prediction on the current canvas content
   *
   * @param {HTMLCanvasElement} canvas - The canvas element
   */
  const handleCanvasChange = (canvas) => {
    // Only make predictions if model is trained
    if (!model) {
      console.log('Model not ready yet. Train the model first!');
      return;
    }

    try {
      // STEP 1: Preprocess the canvas image
      // Converts the drawing to a 28x28 grayscale tensor
      const imageTensor = preprocessCanvas(canvas);

      // STEP 2: Make prediction
      // The model outputs probabilities for each digit (0-9)
      const result = predict(model, imageTensor);

      // STEP 3: Update UI with prediction
      setPrediction(result);

      // STEP 4: Clean up tensor to prevent memory leaks
      // TensorFlow.js requires manual memory management!
      imageTensor.dispose();

      console.log('Prediction:', result.predictedDigit, 'with confidence:', result.confidence + '%');
    } catch (error) {
      console.error('Error making prediction:', error);
    }
  };

  // RENDER
  // ------

  return (
    <div className="App">
      {/* Header Section */}
      <header className="app-header">
        <h1>🚀 React + Deep Learning Tutorial</h1>
        <p>
          Build a handwritten digit recognizer that runs entirely in your browser.
          No backend required - powered by TensorFlow.js!
        </p>
      </header>

      {/* Main Content Grid */}
      <main className="app-content">
        {/*
          COMPONENT 1: Model Trainer
          - Creates the neural network
          - Loads training data
          - Trains the model
          - Calls handleModelReady when done
        */}
        <ModelTrainer onModelReady={handleModelReady} />

        {/*
          COMPONENT 2: Drawing Canvas
          - Provides interactive drawing area
          - Calls handleCanvasChange when user draws
          - This triggers predictions in real-time
        */}
        <div className="card">
          <h2>✏️ Draw Here</h2>
          <DrawingCanvas onCanvasChange={handleCanvasChange} />
        </div>

        {/*
          COMPONENT 3: Prediction Display
          - Shows the predicted digit
          - Displays confidence scores
          - Visualizes probability distribution
        */}
        <PredictionDisplay prediction={prediction} />
      </main>

      {/* Footer with Educational Notes */}
      <footer style={{ marginTop: '3rem', textAlign: 'center', color: '#718096' }}>
        <div className="card" style={{ maxWidth: '800px', margin: '0 auto' }}>
          <h3 style={{ marginBottom: '1rem', color: '#667eea' }}>
            🎓 Learning Objectives
          </h3>
          <div style={{ textAlign: 'left' }}>
            <p style={{ marginBottom: '0.5rem' }}>
              <strong>React Skills:</strong> Component architecture, state management with
              useState, event handling, props and callbacks, useEffect and useRef hooks
            </p>
            <p style={{ marginBottom: '0.5rem' }}>
              <strong>TensorFlow.js Skills:</strong> Building CNN models, training neural
              networks, preprocessing data, making predictions, memory management
            </p>
            <p style={{ marginBottom: '0.5rem' }}>
              <strong>Deep Learning Concepts:</strong> Convolutional layers, pooling, dense
              layers, activation functions (ReLU, Softmax), loss functions, optimizers
            </p>
            <p style={{ marginBottom: '0.5rem' }}>
              <strong>Web APIs:</strong> Canvas API for drawing, event listeners, async/await
              for asynchronous operations
            </p>
          </div>

          <div style={{
            marginTop: '1.5rem',
            paddingTop: '1rem',
            borderTop: '1px solid #e2e8f0',
            fontSize: '0.9rem'
          }}>
            <p>
              <strong>How to Extend This Project:</strong>
            </p>
            <ul style={{ marginTop: '0.5rem', marginLeft: '1.5rem' }}>
              <li>Add data augmentation (rotate, scale, shift images)</li>
              <li>Visualize what the CNN layers learn</li>
              <li>Implement model save/load functionality</li>
              <li>Add support for letter recognition (A-Z)</li>
              <li>Create a multi-digit recognition system</li>
              <li>Add webcam support for real-time recognition</li>
              <li>Implement transfer learning with pre-trained models</li>
            </ul>
          </div>
        </div>

        <p style={{ marginTop: '2rem', fontSize: '0.9rem' }}>
          Made with ❤️ | Powered by React + TensorFlow.js
        </p>
      </footer>
    </div>
  );
}

export default App;
