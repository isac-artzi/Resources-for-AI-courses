/**
 * ResultDisplay Component
 *
 * This component displays the classification results from the backend.
 * It shows:
 * - Top predictions with labels
 * - Confidence scores
 * - Visual representation (progress bars)
 *
 * Key React Concepts:
 * - Props: Receive data from parent component
 * - Conditional Rendering: Show different content based on props
 * - Array.map(): Render lists of items
 */

import React from 'react';
import './ResultDisplay.css';

/**
 * ResultDisplay Component
 *
 * Props:
 *   - result: Object containing prediction results
 *     Format: {
 *       success: true,
 *       predictions: [
 *         { label: "tabby cat", confidence: 0.92, class_id: 281 },
 *         { label: "Egyptian cat", confidence: 0.05, class_id: 285 },
 *         ...
 *       ],
 *       filename: "cat.jpg"
 *     }
 *   - error: String with error message (if something went wrong)
 *
 * Example usage:
 *   <ResultDisplay result={predictionResult} error={errorMessage} />
 */
function ResultDisplay({ result, error }) {

  // ========================================================================
  // CONDITIONAL RENDERING
  // ========================================================================

  /**
   * Case 1: Error occurred
   *
   * Show error message if there's an error
   */
  if (error) {
    return (
      <div className="result-display">
        <div className="error-container">
          {/* Error Icon */}
          <svg
            className="error-icon"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>

          {/* Error Message */}
          <h3 className="error-title">Classification Failed</h3>
          <p className="error-message">{error}</p>

          {/* Help Text */}
          <p className="error-hint">
            Please try again with a different image, or check if the backend is running.
          </p>
        </div>
      </div>
    );
  }

  /**
   * Case 2: No results yet
   *
   * Show nothing if there are no results to display
   */
  if (!result || !result.predictions) {
    return null;
  }

  /**
   * Case 3: Successful prediction
   *
   * Display the classification results
   */

  // Extract data from result object
  const { predictions, filename } = result;

  // ========================================================================
  // HELPER FUNCTIONS
  // ========================================================================

  /**
   * Format confidence as percentage
   *
   * Converts decimal (0.92) to percentage (92%)
   *
   * @param {number} confidence - Confidence value (0-1)
   * @returns {string} Formatted percentage (e.g., "92.5%")
   */
  const formatConfidence = (confidence) => {
    return `${(confidence * 100).toFixed(1)}%`;
  };

  /**
   * Get color for confidence level
   *
   * Returns different colors based on confidence:
   * - High (>0.7): Green
   * - Medium (0.3-0.7): Orange
   * - Low (<0.3): Red
   *
   * @param {number} confidence - Confidence value (0-1)
   * @returns {string} CSS class name
   */
  const getConfidenceColor = (confidence) => {
    if (confidence > 0.7) return 'confidence-high';
    if (confidence > 0.3) return 'confidence-medium';
    return 'confidence-low';
  };

  // ========================================================================
  // RENDER
  // ========================================================================

  return (
    <div className="result-display">
      {/* Header */}
      <div className="result-header">
        {/* Success Icon */}
        <svg
          className="success-icon"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>

        <div>
          <h3 className="result-title">Classification Complete</h3>
          <p className="result-filename">{filename}</p>
        </div>
      </div>

      {/* Predictions List */}
      <div className="predictions-container">
        <h4 className="predictions-title">Top Predictions</h4>

        {/* Map over predictions array to create list items */}
        {/* Array.map() is used to render lists in React */}
        {predictions.map((prediction, index) => (
          <div
            key={prediction.class_id}  // Unique key for React
            className="prediction-item"
          >
            {/* Rank Number */}
            <div className="prediction-rank">
              {index + 1}
            </div>

            {/* Prediction Details */}
            <div className="prediction-content">
              {/* Label and Confidence */}
              <div className="prediction-header">
                <span className="prediction-label">
                  {/* Capitalize first letter */}
                  {prediction.label.charAt(0).toUpperCase() + prediction.label.slice(1)}
                </span>
                <span className={`prediction-confidence ${getConfidenceColor(prediction.confidence)}`}>
                  {formatConfidence(prediction.confidence)}
                </span>
              </div>

              {/* Confidence Bar */}
              <div className="confidence-bar-container">
                <div
                  className={`confidence-bar ${getConfidenceColor(prediction.confidence)}`}
                  style={{
                    width: `${prediction.confidence * 100}%`
                  }}
                >
                  {/* Inner gradient for visual effect */}
                  <div className="confidence-bar-inner"></div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Additional Info */}
      <div className="result-footer">
        <p className="result-info">
          <span className="info-icon">ℹ️</span>
          Predictions are based on the ResNet-18 model trained on ImageNet.
          Confidence scores indicate the model's certainty for each prediction.
        </p>
      </div>
    </div>
  );
}

/**
 * Export the component
 */
export default ResultDisplay;

/**
 * Understanding Array.map() in React:
 *
 * Array.map() is used extensively in React to render lists.
 * It takes an array and returns a new array with transformed items.
 *
 * Example:
 *   const numbers = [1, 2, 3];
 *   const doubled = numbers.map(n => n * 2);
 *   // Result: [2, 4, 6]
 *
 * In React, we use it to transform data into JSX:
 *   predictions.map(pred => <div>{pred.label}</div>)
 *
 * Important: Always provide a unique "key" prop when rendering lists
 * This helps React efficiently update the DOM
 *
 * Key Guidelines:
 * - Use a unique, stable ID (like prediction.class_id)
 * - Don't use array index if the list can be reordered
 * - Keys must be unique among siblings
 */

/**
 * Understanding Conditional Rendering:
 *
 * React components can return different JSX based on conditions.
 * Common patterns:
 *
 * 1. If-Return (used in this component):
 *    if (error) return <Error />;
 *    return <Success />;
 *
 * 2. Ternary Operator:
 *    return error ? <Error /> : <Success />;
 *
 * 3. Logical AND:
 *    return error && <Error />;
 *
 * 4. Switch Statement:
 *    switch(status) {
 *      case 'loading': return <Loading />;
 *      case 'error': return <Error />;
 *      default: return <Success />;
 *    }
 *
 * Choose based on:
 * - Readability
 * - Number of conditions
 * - Complexity of logic
 */
