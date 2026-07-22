/**
 * ModelTrainer.js - Model Training Interface Component
 *
 * This component handles:
 * - Creating the neural network model
 * - Loading training data
 * - Training the model with progress updates
 * - Displaying training metrics
 *
 * React Concepts Used:
 * - useState: Managing training state and progress
 * - useEffect: Loading data when component mounts
 * - Async operations: Handling model training
 */

import React, { useState, useEffect } from 'react';
import { createModel, loadMNISTData, trainModel } from '../utils/modelUtils';

/**
 * ModelTrainer Component
 *
 * @param {Function} onModelReady - Callback when model is trained and ready
 * @returns {JSX.Element}
 */
function ModelTrainer({ onModelReady }) {
  // State for the TensorFlow.js model
  const [model, setModel] = useState(null);

  // State for training data
  const [data, setData] = useState(null);

  // State for training status: 'idle', 'loading', 'ready', 'training', 'trained'
  const [status, setStatus] = useState('idle');

  // State for training progress
  const [progress, setProgress] = useState({
    epoch: 0,
    totalEpochs: 0,
    accuracy: 0,
    valAccuracy: 0,
    loss: 0
  });

  /**
   * Load model and data when component mounts
   */
  useEffect(() => {
    initializeModel();
  }, []); // Run once on mount

  /**
   * Creates model and loads training data
   */
  const initializeModel = async () => {
    try {
      setStatus('loading');

      // Create the neural network
      console.log('Creating model...');
      const newModel = createModel();
      setModel(newModel);

      // Load MNIST training data
      console.log('Loading training data...');
      const mnistData = await loadMNISTData();
      setData(mnistData);

      setStatus('ready');
      console.log('Model and data ready!');
    } catch (error) {
      console.error('Error initializing model:', error);
      setStatus('idle');
      alert('Error loading model or data. Check console for details.');
    }
  };

  /**
   * Starts training the model
   */
  const handleTrainModel = async () => {
    if (!model || !data) {
      alert('Model or data not ready. Please wait...');
      return;
    }

    try {
      setStatus('training');

      // Train the model with progress callback
      await trainModel(
        model,
        data.trainData,
        data.testData,
        (progressInfo) => {
          // Update progress state as training progresses
          setProgress({
            epoch: progressInfo.epoch,
            totalEpochs: progressInfo.totalEpochs,
            accuracy: progressInfo.accuracy,
            valAccuracy: progressInfo.valAccuracy,
            loss: progressInfo.loss
          });
        }
      );

      setStatus('trained');

      // Notify parent component that model is ready for predictions
      if (onModelReady) {
        onModelReady(model);
      }

      console.log('Training complete! Model ready for predictions.');
    } catch (error) {
      console.error('Error training model:', error);
      setStatus('ready');
      alert('Error during training. Check console for details.');
    }
  };

  /**
   * Calculates training progress percentage
   */
  const getProgressPercentage = () => {
    if (progress.totalEpochs === 0) return 0;
    return Math.round((progress.epoch / progress.totalEpochs) * 100);
  };

  /**
   * Gets status badge class
   */
  const getStatusBadgeClass = () => {
    switch (status) {
      case 'training':
        return 'status-training';
      case 'trained':
        return 'status-ready';
      default:
        return 'status-idle';
    }
  };

  /**
   * Gets status text
   */
  const getStatusText = () => {
    switch (status) {
      case 'idle':
        return 'Initializing...';
      case 'loading':
        return 'Loading...';
      case 'ready':
        return 'Ready to Train';
      case 'training':
        return 'Training...';
      case 'trained':
        return 'Model Trained!';
      default:
        return 'Unknown';
    }
  };

  return (
    <div className="card">
      <h2>
        🧠 Model Training
        <span className={`status-badge ${getStatusBadgeClass()}`}>
          {getStatusText()}
        </span>
      </h2>

      <div className="training-controls">
        {/* Training Information */}
        <div className="training-info">
          <p>
            <strong>Model:</strong> Convolutional Neural Network (CNN)
          </p>
          <p>
            <strong>Architecture:</strong> Conv2D → MaxPool → Conv2D → MaxPool → Dense(128) → Dense(10)
          </p>
          <p>
            <strong>Dataset:</strong> MNIST Handwritten Digits
          </p>
        </div>

        {/* Training Button */}
        <button
          className="btn btn-primary"
          onClick={handleTrainModel}
          disabled={status !== 'ready' && status !== 'trained'}
        >
          {status === 'training' && <span className="loading-spinner"></span>}
          {status === 'trained' ? 'Retrain Model' : 'Train Model'}
        </button>

        {/* Progress Bar (shown during training) */}
        {status === 'training' && (
          <div>
            <p style={{ marginTop: '1rem', marginBottom: '0.5rem' }}>
              <strong>Training Progress:</strong> Epoch {progress.epoch} of {progress.totalEpochs}
            </p>
            <div className="progress-bar-container">
              <div
                className="progress-bar"
                style={{ width: `${getProgressPercentage()}%` }}
              >
                {getProgressPercentage()}%
              </div>
            </div>
          </div>
        )}

        {/* Training Metrics (shown during and after training) */}
        {(status === 'training' || status === 'trained') && progress.epoch > 0 && (
          <div className="training-info">
            <h3 style={{ marginTop: '1rem', marginBottom: '0.5rem' }}>
              Training Metrics
            </h3>
            <p>
              <strong>Loss:</strong> {progress.loss.toFixed(4)}
              <br />
              <small style={{ color: '#718096' }}>
                (Lower is better - measures how wrong predictions are)
              </small>
            </p>
            <p>
              <strong>Training Accuracy:</strong> {(progress.accuracy * 100).toFixed(2)}%
              <br />
              <small style={{ color: '#718096' }}>
                (Percentage of correct predictions on training data)
              </small>
            </p>
            <p>
              <strong>Validation Accuracy:</strong> {(progress.valAccuracy * 100).toFixed(2)}%
              <br />
              <small style={{ color: '#718096' }}>
                (Accuracy on unseen data - measures generalization)
              </small>
            </p>
          </div>
        )}

        {/* Success Message */}
        {status === 'trained' && (
          <div className="info-message" style={{ marginTop: '1rem' }}>
            <strong>Success!</strong> Your model is trained and ready. Draw a digit
            on the canvas to see it in action!
          </div>
        )}

        {/* Educational Note */}
        <div className="warning-message" style={{ marginTop: '1rem' }}>
          <strong>Note for Students:</strong> Training happens entirely in your browser!
          The model learns from ~5000 training examples and validates on ~1000 test
          examples. This usually takes 30-60 seconds.
        </div>
      </div>
    </div>
  );
}

export default ModelTrainer;
