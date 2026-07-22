/**
 * Loading Spinner Component
 * =========================
 *
 * A simple loading indicator to show during async operations.
 * Provides visual feedback that processing is happening.
 *
 * Props:
 * - message: Optional message to display below spinner
 *
 * Author: Cloud ML Deployment
 */

function LoadingSpinner({ message = 'Loading...' }) {
  return (
    <div className="loading-spinner">
      {/* Spinner Animation */}
      <div className="spinner">
        <div className="spinner-circle"></div>
        <div className="spinner-circle"></div>
        <div className="spinner-circle"></div>
      </div>

      {/* Loading Message */}
      {message && (
        <p className="loading-message">{message}</p>
      )}
    </div>
  )
}

export default LoadingSpinner

/**
 * Learning Notes:
 * ==============
 *
 * 1. Default Props:
 *    - function Component({ prop = defaultValue })
 *    - Provides fallback if prop not passed
 *    - message = 'Loading...' sets default
 *
 * 2. Conditional Rendering:
 *    - {condition && <Component />}
 *    - Renders only if condition is true
 *    - Short-circuit evaluation
 *
 * 3. Pure Components:
 *    - No state, only props
 *    - Simpler and easier to test
 *    - Can be optimized by React
 *
 * 4. CSS Animation:
 *    - Animation defined in CSS
 *    - Component just provides structure
 *    - Separation of concerns
 */
