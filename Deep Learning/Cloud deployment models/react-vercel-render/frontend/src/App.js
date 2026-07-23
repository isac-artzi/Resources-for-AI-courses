/**
 * Main App Component
 *
 * This is the root component for the Deep Learning Image Classification app.
 * It handles image upload, API communication, and result display.
 */

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import ImageUpload from './components/ImageUpload';
import Results from './components/Results';
import { ENDPOINTS, CONFIG } from './config';

function App() {
  // State management
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');

  // Check API health on component mount
  useEffect(() => {
    checkApiHealth();
  }, []);

  /**
   * Check if the backend API is healthy
   */
  const checkApiHealth = async () => {
    try {
      const response = await axios.get(ENDPOINTS.health);
      if (response.data.status === 'healthy') {
        setApiStatus('connected');
      } else {
        setApiStatus('error');
      }
    } catch (err) {
      console.error('API health check failed:', err);
      setApiStatus('disconnected');
    }
  };

  /**
   * Handle file selection
   * @param {Event} event - File input change event
   */
  const handleFileSelect = (event) => {
    const file = event.target.files[0];

    // Reset previous state
    setError(null);
    setPredictions(null);

    if (!file) {
      setSelectedFile(null);
      setPreview(null);
      return;
    }

    // Validate file type
    if (!CONFIG.allowedFileTypes.includes(file.type)) {
      setError('Please select a valid image file (JPEG, PNG, WebP)');
      return;
    }

    // Validate file size
    if (file.size > CONFIG.maxFileSize) {
      setError(`File size must be less than ${CONFIG.maxFileSize / (1024 * 1024)}MB`);
      return;
    }

    // Set file and create preview
    setSelectedFile(file);
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result);
    };
    reader.readAsDataURL(file);
  };

  /**
   * Handle form submission - send image to backend
   * @param {Event} event - Form submit event
   */
  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);
    setPredictions(null);

    try {
      // Create FormData to send file
      const formData = new FormData();
      formData.append('file', selectedFile);

      // Send request to backend
      const response = await axios.post(ENDPOINTS.predict, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000, // 30 second timeout
      });

      // Handle successful response
      if (response.data.success) {
        setPredictions(response.data.predictions);
      } else {
        setError('Prediction failed. Please try again.');
      }
    } catch (err) {
      console.error('Prediction error:', err);

      // Handle different error types
      if (err.code === 'ECONNABORTED') {
        setError('Request timeout. The server took too long to respond.');
      } else if (err.response) {
        // Server responded with error
        setError(err.response.data.detail || 'Server error occurred');
      } else if (err.request) {
        // Request made but no response
        setError('Cannot connect to server. Please check if the backend is running.');
      } else {
        // Other errors
        setError('An unexpected error occurred');
      }
    } finally {
      setLoading(false);
    }
  };

  /**
   * Clear all selections and results
   */
  const handleClear = () => {
    setSelectedFile(null);
    setPreview(null);
    setPredictions(null);
    setError(null);
  };

  return (
    <div className="App">
      {/* Header */}
      <header className="App-header">
        <h1>Deep Learning Image Classifier</h1>
        <p>Upload an image to classify it using AI</p>

        {/* API Status Indicator */}
        <div className={`api-status ${apiStatus}`}>
          <span className="status-dot"></span>
          {apiStatus === 'connected' && 'API Connected'}
          {apiStatus === 'disconnected' && 'API Disconnected'}
          {apiStatus === 'checking' && 'Checking API...'}
          {apiStatus === 'error' && 'API Error'}
        </div>
      </header>

      {/* Main Content */}
      <main className="App-main">
        <div className="container">

          {/* Upload Section */}
          <ImageUpload
            preview={preview}
            selectedFile={selectedFile}
            onFileSelect={handleFileSelect}
            onSubmit={handleSubmit}
            onClear={handleClear}
            loading={loading}
          />

          {/* Error Display */}
          {error && (
            <div className="error-message">
              <span className="error-icon">⚠️</span>
              {error}
            </div>
          )}

          {/* Loading Indicator */}
          {loading && (
            <div className="loading">
              <div className="spinner"></div>
              <p>Analyzing image...</p>
            </div>
          )}

          {/* Results Display */}
          {predictions && !loading && (
            <Results predictions={predictions} />
          )}

          {/* Instructions */}
          {!selectedFile && !loading && (
            <div className="instructions">
              <h3>How to use:</h3>
              <ol>
                <li>Click "Choose Image" to select an image file</li>
                <li>Supported formats: JPEG, PNG, WebP</li>
                <li>Maximum file size: 10MB</li>
                <li>Click "Classify Image" to get predictions</li>
              </ol>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="App-footer">
        <p>Powered by TensorFlow & FastAPI</p>
        <p>Frontend: React on Vercel | Backend: FastAPI on Render</p>
      </footer>
    </div>
  );
}

export default App;
