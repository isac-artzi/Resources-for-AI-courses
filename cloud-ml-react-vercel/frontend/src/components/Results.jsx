/**
 * Results Display Component
 * =========================
 *
 * This component displays the classification results
 * in a clean, user-friendly format.
 *
 * Features:
 * - Top prediction highlight
 * - Confidence scores with visual bars
 * - Color-coded confidence levels
 * - Responsive design
 *
 * Props:
 * - predictions: Array of prediction objects
 *
 * Author: Cloud ML Deployment
 */

function Results({ predictions }) {
  // =========================================================================
  // Helper Functions
  // =========================================================================

  /**
   * Get color class based on confidence level
   * High confidence = green, medium = yellow, low = red
   *
   * @param {number} confidence - Confidence score (0-1)
   * @returns {string} - CSS class name
   */
  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.7) return 'confidence-high'
    if (confidence >= 0.4) return 'confidence-medium'
    return 'confidence-low'
  }

  /**
   * Get medal emoji for top 3 predictions
   *
   * @param {number} rank - Position in results (1-5)
   * @returns {string} - Medal emoji or empty string
   */
  const getMedalEmoji = (rank) => {
    switch (rank) {
      case 1: return '🥇'
      case 2: return '🥈'
      case 3: return '🥉'
      default: return ''
    }
  }

  // =========================================================================
  // Validation
  // =========================================================================

  // Check if predictions exist and are valid
  if (!predictions || predictions.length === 0) {
    return (
      <div className="results">
        <h2>📊 Results</h2>
        <p className="no-results">No predictions available</p>
      </div>
    )
  }

  // =========================================================================
  // Render
  // =========================================================================

  // Get the top prediction (highest confidence)
  const topPrediction = predictions[0]

  return (
    <div className="results">
      <h2>📊 Classification Results</h2>

      {/* Top Prediction Highlight */}
      <div className="top-prediction">
        <div className="prediction-label">
          🎯 Most Likely:
        </div>
        <div className="prediction-value">
          {topPrediction.class}
        </div>
        <div className="prediction-confidence">
          {topPrediction.confidence_percent} confident
        </div>
      </div>

      {/* All Predictions List */}
      <div className="predictions-list">
        <h3>Top {predictions.length} Predictions</h3>

        {predictions.map((prediction, index) => {
          // Calculate percentage for progress bar width
          const percentage = prediction.confidence * 100

          return (
            <div
              key={prediction.class_id}
              className={`
                prediction-item
                ${index === 0 ? 'top-item' : ''}
              `}
            >
              {/* Rank and Medal */}
              <div className="prediction-rank">
                <span className="rank-number">{prediction.rank}</span>
                <span className="rank-medal">{getMedalEmoji(prediction.rank)}</span>
              </div>

              {/* Prediction Details */}
              <div className="prediction-details">
                {/* Class Name */}
                <div className="prediction-class">
                  {prediction.class}
                </div>

                {/* Confidence Bar */}
                <div className="confidence-bar-container">
                  {/* Background bar */}
                  <div className="confidence-bar-bg">
                    {/* Filled bar */}
                    <div
                      className={`
                        confidence-bar-fill
                        ${getConfidenceColor(prediction.confidence)}
                      `}
                      style={{ width: `${percentage}%` }}
                    >
                      {/* Confidence text inside bar */}
                      <span className="confidence-text">
                        {prediction.confidence_percent}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Additional Information */}
      <div className="results-info">
        <p className="info-text">
          ℹ️ Predictions are based on MobileNetV2 trained on ImageNet
        </p>
        <p className="info-text">
          🎓 Model recognizes 1,000 different object categories
        </p>
      </div>
    </div>
  )
}

export default Results

/**
 * Learning Notes:
 * ==============
 *
 * 1. Array Mapping:
 *    - predictions.map() creates JSX for each prediction
 *    - Each item needs a unique 'key' prop
 *    - Keys help React identify which items changed
 *
 * 2. Inline Styles:
 *    - style={{ property: value }}
 *    - Double curly braces: outer for JSX, inner for object
 *    - Useful for dynamic values (like width percentage)
 *
 * 3. Conditional Rendering:
 *    - Early return pattern for validation
 *    - Renders different content based on data
 *
 * 4. CSS Classes:
 *    - Multiple classes with template literals
 *    - Dynamic classes based on conditions
 *    - Helps with styling and animations
 *
 * 5. Helper Functions:
 *    - Keep render logic clean
 *    - Reusable logic extraction
 *    - Easier to test and maintain
 *
 * 6. Destructuring:
 *    - Extract specific properties from objects
 *    - Cleaner code, less repetition
 *    - const { class } = prediction
 */
