/**
 * Results Component
 *
 * Displays classification predictions with confidence scores
 */

import React from 'react';
import './Results.css';

function Results({ predictions }) {
  /**
   * Format class name for display
   * @param {string} className - Raw class name from model
   * @returns {string} Formatted class name
   */
  const formatClassName = (className) => {
    return className
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  /**
   * Get confidence bar color based on confidence level
   * @param {number} confidence - Confidence score (0-1)
   * @returns {string} CSS class name for color
   */
  const getConfidenceColor = (confidence) => {
    if (confidence > 0.7) return 'high';
    if (confidence > 0.4) return 'medium';
    return 'low';
  };

  return (
    <div className="results-section">
      <h2 className="results-title">Classification Results</h2>

      <div className="predictions-list">
        {predictions.map((prediction, index) => (
          <div key={index} className="prediction-item">

            {/* Rank Badge */}
            <div className="rank-badge">#{index + 1}</div>

            {/* Class Name */}
            <div className="prediction-content">
              <div className="class-name">
                {formatClassName(prediction.class)}
              </div>

              {/* Confidence Bar */}
              <div className="confidence-bar-container">
                <div
                  className={`confidence-bar ${getConfidenceColor(prediction.confidence)}`}
                  style={{ width: `${prediction.confidence * 100}%` }}
                >
                  <span className="confidence-text">
                    {(prediction.confidence * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Top Prediction Highlight */}
      {predictions.length > 0 && (
        <div className="top-prediction">
          <p className="top-label">Most likely:</p>
          <p className="top-class">
            {formatClassName(predictions[0].class)}
          </p>
          <p className="top-confidence">
            Confidence: {(predictions[0].confidence * 100).toFixed(1)}%
          </p>
        </div>
      )}
    </div>
  );
}

export default Results;
