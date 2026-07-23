/**
 * PredictionDisplay.js - Prediction Results Component
 *
 * This component displays:
 * - The predicted digit
 * - Confidence score
 * - Probability distribution for all digits (0-9)
 *
 * React Concepts Used:
 * - Props: Receiving prediction data from parent
 * - Conditional rendering: Showing different UI based on state
 * - Array mapping: Rendering lists of items
 */

import React from 'react';

/**
 * PredictionDisplay Component
 *
 * @param {Object} prediction - Prediction results from the model
 * @param {number} prediction.predictedDigit - The digit with highest probability
 * @param {string} prediction.confidence - Confidence percentage
 * @param {Array} prediction.probabilities - Array of probabilities for each digit
 * @returns {JSX.Element}
 */
function PredictionDisplay({ prediction }) {
  /**
   * Determines if this prediction is empty (no drawing)
   */
  const isEmpty = !prediction || prediction.confidence === '0.00';

  return (
    <div className="card">
      <h2>🎯 Predictions</h2>

      {/* Show message if no prediction yet */}
      {isEmpty ? (
        <div className="info-message">
          <strong>No prediction yet.</strong>
          <br />
          Train the model, then draw a digit on the canvas to see predictions!
        </div>
      ) : (
        <>
          {/* Main Prediction Result */}
          <div className="prediction-result">
            <h3>Predicted Digit:</h3>
            <p className="prediction-digit">{prediction.predictedDigit}</p>
            <p className="prediction-confidence">
              Confidence: {prediction.confidence}%
            </p>
          </div>

          {/* Probability Distribution */}
          <div style={{ marginTop: '2rem' }}>
            <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem' }}>
              Probability Distribution
            </h3>
            <p style={{ color: '#718096', fontSize: '0.9rem', marginBottom: '1rem' }}>
              These bars show how confident the model is for each possible digit.
              The longer the bar, the more confident the model is.
            </p>

            {/* Render a bar for each digit (0-9) */}
            <div className="prediction-grid">
              {prediction.probabilities.map((item) => (
                <div key={item.digit} className="prediction-item">
                  {/* Digit Label */}
                  <span className="prediction-label">{item.digit}</span>

                  {/* Probability Bar */}
                  <div className="prediction-bar-container">
                    <div
                      className={`prediction-bar ${
                        item.digit === prediction.predictedDigit
                          ? 'top-prediction'
                          : ''
                      }`}
                      style={{
                        width: `${item.probability * 100}%`
                      }}
                    />
                  </div>

                  {/* Percentage Value */}
                  <span className="prediction-value">{item.percentage}%</span>
                </div>
              ))}
            </div>
          </div>

          {/* Educational Note */}
          <div className="info-message" style={{ marginTop: '1.5rem' }}>
            <strong>Understanding the Results:</strong>
            <ul style={{ marginTop: '0.5rem', marginLeft: '1.5rem' }}>
              <li>
                The <strong>predicted digit</strong> is the one with the highest probability
              </li>
              <li>
                <strong>Confidence</strong> shows how sure the model is (higher is better)
              </li>
              <li>
                All probabilities add up to 100% (thanks to the Softmax activation!)
              </li>
            </ul>
          </div>
        </>
      )}
    </div>
  );
}

export default PredictionDisplay;
