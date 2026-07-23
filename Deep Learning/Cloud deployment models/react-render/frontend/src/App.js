/**
 * Main App Component
 *
 * This is the root component of our React application.
 * It coordinates all other components and manages the overall state.
 *
 * Key Responsibilities:
 * - Manage application state (uploaded image, results, loading, errors)
 * - Handle image upload to backend
 * - Coordinate communication between components
 * - Display ImageUpload and ResultDisplay components
 *
 * React Concepts Demonstrated:
 * - useState Hook: Managing component state
 * - useEffect Hook: Side effects (API health check)
 * - Component Composition: Using child components
 * - Async/Await: Handling asynchronous operations
 * - Event Handling: Responding to user actions
 */

import React, { useState, useEffect } from 'react';
import './App.css';

// Import our components
import ImageUpload from './components/ImageUpload';
import ResultDisplay from './components/ResultDisplay';

// Import API service
import api from './services/api';

/**
 * Main App Component
 *
 * This component doesn't receive any props - it's the top-level component
 */
function App() {

  // ==========================================================================
  // STATE MANAGEMENT
  // ==========================================================================

  /**
   * State: loading
   *
   * Tracks if an upload/prediction is in progress
   * - true: Show loading indicators, disable buttons
   * - false: Normal state
   */
  const [loading, setLoading] = useState(false);

  /**
   * State: result
   *
   * Stores the prediction results from the backend
   * - null: No results yet
   * - Object: Contains predictions, filename, etc.
   */
  const [result, setResult] = useState(null);

  /**
   * State: error
   *
   * Stores error messages if something goes wrong
   * - null: No error
   * - String: Error message to display
   */
  const [error, setError] = useState(null);

  /**
   * State: backendStatus
   *
   * Tracks if backend is reachable
   * - 'unknown': Haven't checked yet
   * - 'online': Backend is responding
   * - 'offline': Backend is not reachable
   */
  const [backendStatus, setBackendStatus] = useState('unknown');

  // ==========================================================================
  // EFFECTS (Side Effects)
  // ==========================================================================

  /**
   * useEffect: Check backend health on component mount
   *
   * This runs once when the component first renders
   * It checks if the backend API is online and ready
   *
   * Why useEffect?
   * - Side effects (like API calls) shouldn't happen during render
   * - useEffect runs AFTER the component renders
   * - The empty array [] means "run only once on mount"
   */
  useEffect(() => {
    // Define an async function inside useEffect
    // (useEffect itself can't be async)
    const checkBackend = async () => {
      console.log('Checking backend health...');

      const health = await api.checkHealth();

      if (health && health.status === 'healthy') {
        setBackendStatus('online');
        console.log('✅ Backend is online and ready!');
      } else {
        setBackendStatus('offline');
        console.warn('⚠️ Backend is offline or not responding');
      }
    };

    // Call the async function
    checkBackend();

    // Cleanup function (optional)
    // Runs when component unmounts
    return () => {
      console.log('App component unmounting');
    };
  }, []); // Empty dependency array = run once on mount

  // ==========================================================================
  // EVENT HANDLERS
  // ==========================================================================

  /**
   * Handle image upload
   *
   * This function is called when the user clicks "Classify Image"
   * It receives the selected image file from the ImageUpload component
   *
   * Flow:
   * 1. Reset previous results and errors
   * 2. Set loading state
   * 3. Call API to upload image
   * 4. Handle response (success or error)
   * 5. Clear loading state
   *
   * @param {File} imageFile - The image file to upload
   */
  const handleImageUpload = async (imageFile) => {
    // ----------------------------------------------------------------
    // STEP 1: Reset state and validate
    // ----------------------------------------------------------------

    console.log('Starting image upload...');

    // Clear any previous results or errors
    setResult(null);
    setError(null);

    // Set loading state (shows spinner, disables buttons)
    setLoading(true);

    try {
      // ----------------------------------------------------------------
      // STEP 2: Upload image to backend
      // ----------------------------------------------------------------

      console.log('Uploading image to backend...');

      // Call the API service to upload the image
      // This is an async operation, so we use 'await'
      const response = await api.uploadImage(imageFile);

      // ----------------------------------------------------------------
      // STEP 3: Handle the response
      // ----------------------------------------------------------------

      // Check if the upload was successful
      if (response.success) {
        // SUCCESS: Store the results
        console.log('✅ Classification successful!');
        console.log('Predictions:', response.predictions);

        setResult(response);
        setError(null);

        // Scroll to results (smooth user experience)
        setTimeout(() => {
          document.querySelector('.result-display')?.scrollIntoView({
            behavior: 'smooth',
            block: 'nearest'
          });
        }, 100);

      } else {
        // ERROR: Something went wrong
        console.error('❌ Classification failed:', response.error);
        setError(response.error || 'Classification failed');
        setResult(null);
      }

    } catch (err) {
      // ----------------------------------------------------------------
      // STEP 4: Handle unexpected errors
      // ----------------------------------------------------------------

      console.error('❌ Unexpected error during upload:', err);

      // Set a user-friendly error message
      setError(
        'An unexpected error occurred. Please check if the backend is running.'
      );
      setResult(null);

    } finally {
      // ----------------------------------------------------------------
      // STEP 5: Cleanup (always runs, even if error occurred)
      // ----------------------------------------------------------------

      // Clear loading state
      setLoading(false);
      console.log('Upload process complete');
    }
  };

  /**
   * Clear results and start over
   *
   * This allows users to classify another image
   */
  const handleReset = () => {
    setResult(null);
    setError(null);
    setLoading(false);
  };

  // ==========================================================================
  // RENDER
  // ==========================================================================

  return (
    <div className="App">
      {/* Main Container */}
      <div className="app-container">

        {/* Header */}
        <header className="app-header">
          <h1 className="app-title">
            🤖 AI Image Classifier
          </h1>
          <p className="app-description">
            Upload any image and our deep learning model will identify what's in it.
            Powered by PyTorch ResNet-18 trained on ImageNet.
          </p>

          {/* Backend Status Indicator */}
          <div className={`status-badge status-${backendStatus}`}>
            <span className="status-dot"></span>
            {backendStatus === 'online' && 'Backend Online'}
            {backendStatus === 'offline' && 'Backend Offline'}
            {backendStatus === 'unknown' && 'Checking Backend...'}
          </div>
        </header>

        {/* Main Content */}
        <main className="app-main">
          {/* Image Upload Section */}
          <ImageUpload
            onUpload={handleImageUpload}
            isLoading={loading}
          />

          {/* Results Section */}
          {/* Only show if we have results or an error */}
          {(result || error) && (
            <ResultDisplay
              result={result}
              error={error}
            />
          )}

          {/* Reset Button (shown after results) */}
          {result && (
            <div className="reset-section">
              <button
                onClick={handleReset}
                className="btn-reset"
              >
                Classify Another Image
              </button>
            </div>
          )}
        </main>

        {/* Footer */}
        <footer className="app-footer">
          <p>
            Built with ❤️ using React & FastAPI
          </p>
          <p className="footer-info">
            Cloud ML Deployment Tutorial Project
          </p>
        </footer>
      </div>
    </div>
  );
}

/**
 * Export the App component
 *
 * This makes it available for import in index.js
 */
export default App;

/**
 * LEARNING NOTES FOR STUDENTS
 * ============================================================================
 *
 * 1. STATE MANAGEMENT:
 *    - useState creates state variables that cause re-renders when updated
 *    - Each setState call triggers a re-render
 *    - State updates are asynchronous
 *    - Never modify state directly (always use setState)
 *
 *    Example:
 *      // ❌ Wrong
 *      result.predictions = [...];
 *
 *      // ✅ Correct
 *      setResult({...result, predictions: [...]});
 *
 * 2. COMPONENT LIFECYCLE:
 *    - Mount: Component is created and added to DOM
 *      → useEffect with [] runs
 *    - Update: State or props change
 *      → Component re-renders
 *    - Unmount: Component is removed from DOM
 *      → useEffect cleanup function runs
 *
 * 3. ASYNC OPERATIONS:
 *    - Always use try-catch with async/await
 *    - Set loading states before async operations
 *    - Clear loading states in finally block
 *    - Handle both success and error cases
 *
 * 4. COMPONENT COMPOSITION:
 *    - Break UI into small, reusable components
 *    - Pass data down via props
 *    - Pass functions down to allow child components to communicate up
 *    - Keep components focused on one responsibility
 *
 * 5. USER EXPERIENCE:
 *    - Show loading indicators during operations
 *    - Display clear error messages
 *    - Provide visual feedback for actions
 *    - Make the interface intuitive
 *
 * 6. DEBUGGING TIPS:
 *    - Use console.log liberally
 *    - Check React DevTools for state/props
 *    - Check Network tab for API calls
 *    - Check Console for errors
 *
 * ============================================================================
 */
