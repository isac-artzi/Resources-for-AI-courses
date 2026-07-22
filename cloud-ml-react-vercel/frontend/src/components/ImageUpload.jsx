/**
 * Image Upload Component
 * ======================
 *
 * This component handles:
 * - File input for image selection
 * - Drag and drop functionality
 * - Image preview display
 * - File validation
 *
 * Props:
 * - onImageSelect: Callback function when image is selected
 * - preview: URL of the image preview
 * - disabled: Whether upload is disabled (e.g., during loading)
 *
 * Author: Cloud ML Deployment
 */

import { useState, useRef } from 'react'

function ImageUpload({ onImageSelect, preview, disabled }) {
  // =========================================================================
  // State Management
  // =========================================================================

  // Track drag state for visual feedback
  const [isDragging, setIsDragging] = useState(false)

  // Reference to the file input element
  // useRef creates a mutable reference that persists across renders
  const fileInputRef = useRef(null)

  // =========================================================================
  // Configuration
  // =========================================================================

  // Allowed file types
  const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif']

  // Maximum file size (10MB in bytes)
  const MAX_FILE_SIZE = 10 * 1024 * 1024

  // =========================================================================
  // File Validation
  // =========================================================================

  /**
   * Validate selected file
   * Checks file type and size
   *
   * @param {File} file - The file to validate
   * @returns {Object} - { valid: boolean, error: string }
   */
  const validateFile = (file) => {
    // Check if file exists
    if (!file) {
      return { valid: false, error: 'No file selected' }
    }

    // Check file type
    if (!ALLOWED_TYPES.includes(file.type)) {
      return {
        valid: false,
        error: `Invalid file type. Allowed: ${ALLOWED_TYPES.join(', ')}`
      }
    }

    // Check file size
    if (file.size > MAX_FILE_SIZE) {
      return {
        valid: false,
        error: `File too large. Maximum size: ${MAX_FILE_SIZE / 1024 / 1024}MB`
      }
    }

    return { valid: true }
  }

  // =========================================================================
  // Event Handlers
  // =========================================================================

  /**
   * Handle file selection from input
   * Called when user selects a file using the file picker
   */
  const handleFileChange = (e) => {
    const file = e.target.files[0]

    if (file) {
      const validation = validateFile(file)

      if (validation.valid) {
        onImageSelect(file)
      } else {
        alert(validation.error)
        // Reset input
        e.target.value = ''
      }
    }
  }

  /**
   * Handle drag enter
   * Called when dragged item enters drop zone
   */
  const handleDragEnter = (e) => {
    e.preventDefault()
    e.stopPropagation()

    if (!disabled) {
      setIsDragging(true)
    }
  }

  /**
   * Handle drag leave
   * Called when dragged item leaves drop zone
   */
  const handleDragLeave = (e) => {
    e.preventDefault()
    e.stopPropagation()

    setIsDragging(false)
  }

  /**
   * Handle drag over
   * Called continuously while item is dragged over drop zone
   * Must prevent default to allow dropping
   */
  const handleDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
  }

  /**
   * Handle drop
   * Called when user drops a file in the drop zone
   */
  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()

    setIsDragging(false)

    if (disabled) return

    // Get the dropped files
    // DataTransfer API provides access to drag/drop data
    const files = e.dataTransfer.files

    if (files && files.length > 0) {
      const file = files[0]
      const validation = validateFile(file)

      if (validation.valid) {
        onImageSelect(file)
      } else {
        alert(validation.error)
      }
    }
  }

  /**
   * Handle click on drop zone
   * Triggers the hidden file input
   */
  const handleClick = () => {
    if (!disabled && fileInputRef.current) {
      fileInputRef.current.click()
    }
  }

  // =========================================================================
  // Render
  // =========================================================================

  return (
    <div className="image-upload">
      <h2>📤 Upload Image</h2>

      {/* Drop Zone */}
      <div
        className={`
          drop-zone
          ${isDragging ? 'dragging' : ''}
          ${disabled ? 'disabled' : ''}
          ${preview ? 'has-preview' : ''}
        `}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={handleClick}
      >
        {/* Preview or Instructions */}
        {preview ? (
          <div className="preview-container">
            <img
              src={preview}
              alt="Preview"
              className="preview-image"
            />
            <p className="preview-text">
              Click or drag to change image
            </p>
          </div>
        ) : (
          <div className="upload-instructions">
            <div className="upload-icon">📸</div>
            <p className="upload-text">
              {isDragging
                ? 'Drop image here'
                : 'Click to select or drag & drop an image'
              }
            </p>
            <p className="upload-subtext">
              Supported: JPG, PNG, GIF (max 10MB)
            </p>
          </div>
        )}

        {/* Hidden File Input */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          disabled={disabled}
          style={{ display: 'none' }}
        />
      </div>

      {/* File Info */}
      {preview && (
        <div className="file-info">
          <p>✅ Image ready for classification</p>
        </div>
      )}
    </div>
  )
}

export default ImageUpload

/**
 * Learning Notes:
 * ==============
 *
 * 1. useRef Hook:
 *    - Creates a mutable reference to DOM elements
 *    - Persists across re-renders
 *    - Doesn't trigger re-renders when changed
 *    - Access DOM element: fileInputRef.current
 *
 * 2. Drag and Drop Events:
 *    - dragenter: Item enters drop zone
 *    - dragleave: Item leaves drop zone
 *    - dragover: Item is over drop zone
 *    - drop: Item is dropped
 *    - Must preventDefault() to allow dropping
 *
 * 3. File Input:
 *    - <input type="file"> for file selection
 *    - Hidden with display:none, triggered via ref.click()
 *    - Access files via e.target.files
 *
 * 4. Conditional CSS Classes:
 *    - Template literals for dynamic classes
 *    - ${condition ? 'class' : ''} pattern
 *    - Multiple classes in one string
 *
 * 5. Props:
 *    - Functions passed as props (onImageSelect)
 *    - Called with specific data (file)
 *    - Parent component receives the data
 */
