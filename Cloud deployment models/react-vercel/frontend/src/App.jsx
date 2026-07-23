/**
 * Main Application Component
 * ==========================
 *
 * This is the root component of the React application.
 * It manages the overall application state and coordinates
 * communication between child components.
 *
 * State Management:
 * - selectedImage: The image file selected by the user
 * - imagePreview: URL for displaying the image preview
 * - predictions: Classification results from the API
 * - loading: Whether a prediction is in progress
 * - error: Error message if something goes wrong
 *
 * Author: Cloud ML Deployment
 */

import { useState } from 'react'
import ImageUpload from './components/ImageUpload'
import Results from './components/Results'
import LoadingSpinner from './components/LoadingSpinner'
import { classifyImage } from './services/api'
import './App.css'

function App() {
  // =========================================================================
  // State Management with useState Hook
  // =========================================================================
  // useState creates state variables that persist across re-renders
  // Format: const [value, setValue] = useState(initialValue)

  // Store the selected image file
  const [selectedImage, setSelectedImage] = useState(null)

  // Store the image preview URL (for displaying the image)
  const [imagePreview, setImagePreview] = useState(null)

  // Store prediction results from the API
  const [predictions, setPredictions] = useState(null)

  // Track loading state (true while waiting for API response)
  const [loading, setLoading] = useState(false)

  // Store error messages
  const [error, setError] = useState(null)

  // =========================================================================
  // Event Handlers
  // =========================================================================

  /**
   * Handle image selection
   * Called when user selects an image file
   *
   * @param {File} file - The selected image file
   */
  const handleImageSelect = (file) => {
    // Clear previous results and errors
    setPredictions(null)
    setError(null)

    // Store the selected file
    setSelectedImage(file)

    // Create a preview URL for the image
    // URL.createObjectURL creates a temporary URL pointing to the file
    const previewUrl = URL.createObjectURL(file)
    setImagePreview(previewUrl)

    console.log('Image selected:', file.name, file.type, file.size)
  }

  /**
   * Handle image classification
   * Sends the image to the API and displays results
   */
  const handleClassify = async () => {
    // Validate that an image is selected
    if (!selectedImage) {
      setError('Please select an image first')
      return
    }

    // Reset error state
    setError(null)

    // Set loading state to show spinner
    setLoading(true)

    try {
      console.log('Sending image to API...')

      // =====================================================================
      // Call API to classify the image
      // =====================================================================
      // classifyImage is an async function that returns a Promise
      // We use 'await' to wait for the API response
      const result = await classifyImage(selectedImage)

      console.log('Received predictions:', result)

      // =====================================================================
      // Update state with predictions
      // =====================================================================
      setPredictions(result.predictions)

      // Log success metrics
      console.log(
        `Classification complete in ${result.processing_time}s`,
        `Top prediction: ${result.predictions[0].class}`,
        `Confidence: ${result.predictions[0].confidence_percent}`
      )

    } catch (err) {
      // =====================================================================
      // Error Handling
      // =====================================================================
      console.error('Classification error:', err)

      // Set user-friendly error message
      if (err.message.includes('Network')) {
        setError('Cannot connect to server. Make sure the backend is running.')
      } else if (err.message.includes('timeout')) {
        setError('Request timed out. Please try again.')
      } else {
        setError(err.message || 'Failed to classify image')
      }
    } finally {
      // =====================================================================
      // Cleanup
      // =====================================================================
      // 'finally' runs regardless of success or failure
      // Always set loading to false when done
      setLoading(false)
    }
  }

  /**
   * Handle reset
   * Clear all state and start over
   */
  const handleReset = () => {
    // Clean up the preview URL to free memory
    if (imagePreview) {
      URL.revokeObjectURL(imagePreview)
    }

    // Reset all state to initial values
    setSelectedImage(null)
    setImagePreview(null)
    setPredictions(null)
    setError(null)
    setLoading(false)

    console.log('Application reset')
  }

  // =========================================================================
  // Render UI
  // =========================================================================
  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <h1>🤖 AI Image Classifier</h1>
        <p className="subtitle">
          Powered by FastAPI + React + TensorFlow
        </p>
        <p className="course-tag">Cloud ML Deployment</p>
      </header>

      {/* Main Content */}
      <main className="app-main">
        <div className="container">

          {/* Instructions Card */}
          <div className="card instructions">
            <h2>📋 How to Use</h2>
            <ol>
              <li>Upload or drag & drop an image (JPG, PNG)</li>
              <li>Click "Classify Image" to analyze</li>
              <li>View predictions with confidence scores</li>
            </ol>
          </div>

          {/* Image Upload Section */}
          <div className="card">
            <ImageUpload
              onImageSelect={handleImageSelect}
              preview={imagePreview}
              disabled={loading}
            />
          </div>

          {/* Action Buttons */}
          <div className="button-group">
            <button
              className="btn btn-primary"
              onClick={handleClassify}
              disabled={!selectedImage || loading}
            >
              {loading ? 'Classifying...' : '🔍 Classify Image'}
            </button>

            <button
              className="btn btn-secondary"
              onClick={handleReset}
              disabled={loading}
            >
              🔄 Reset
            </button>
          </div>

          {/* Loading Spinner */}
          {loading && (
            <div className="card">
              <LoadingSpinner message="Analyzing image with AI..." />
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="card error-card">
              <h3>❌ Error</h3>
              <p>{error}</p>
              <button className="btn btn-secondary" onClick={handleReset}>
                Try Again
              </button>
            </div>
          )}

          {/* Results */}
          {predictions && !loading && (
            <div className="card">
              <Results predictions={predictions} />
            </div>
          )}

        </div>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>
          Built with ❤️ |
          <a href="https://github.com" target="_blank" rel="noopener noreferrer">
            {' '}View on GitHub
          </a>
        </p>
      </footer>
    </div>
  )
}

export default App

/**
 * Learning Notes for Students:
 * ============================
 *
 * 1. Component Structure:
 *    - Imports at top
 *    - Component function
 *    - State management with hooks
 *    - Event handlers
 *    - Return JSX
 *
 * 2. React Hooks Used:
 *    - useState: Manage component state
 *    - All hooks start with 'use'
 *
 * 3. Async/Await:
 *    - async functions return Promises
 *    - await pauses execution until Promise resolves
 *    - try/catch/finally for error handling
 *
 * 4. State Updates:
 *    - Never modify state directly
 *    - Always use setState functions
 *    - React re-renders when state changes
 *
 * 5. Conditional Rendering:
 *    - {condition && <Component />} - renders if condition is true
 *    - Common pattern for showing/hiding elements
 */
