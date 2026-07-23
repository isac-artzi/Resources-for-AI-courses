/**
 * ImageUpload Component
 *
 * This component handles image selection and upload.
 * It provides a user interface for:
 * - Selecting an image file
 * - Previewing the selected image
 * - Uploading to the backend
 *
 * Key React Concepts:
 * - useState: Store component state (selected image, preview)
 * - Props: Receive data from parent (onUpload, isLoading)
 * - Event Handlers: Respond to user actions (file selection, button clicks)
 */

import React, { useState } from 'react';
import './ImageUpload.css';

/**
 * ImageUpload Component
 *
 * Props:
 *   - onUpload: Function to call when user uploads an image
 *   - isLoading: Boolean indicating if upload is in progress
 *
 * Example usage:
 *   <ImageUpload
 *     onUpload={(file) => handleUpload(file)}
 *     isLoading={uploading}
 *   />
 */
function ImageUpload({ onUpload, isLoading }) {

  // ========================================================================
  // STATE MANAGEMENT
  // ========================================================================

  /**
   * State: selectedFile
   *
   * Stores the file object when user selects an image
   * Initial value: null (no file selected)
   *
   * Usage:
   *   - selectedFile: Current value
   *   - setSelectedFile: Function to update value
   */
  const [selectedFile, setSelectedFile] = useState(null);

  /**
   * State: previewUrl
   *
   * Stores the URL for previewing the selected image
   * Initial value: null (no preview)
   *
   * We create a temporary URL using URL.createObjectURL()
   * This URL only exists in the browser, not on any server
   */
  const [previewUrl, setPreviewUrl] = useState(null);

  /**
   * State: dragActive
   *
   * Tracks if user is dragging a file over the drop zone
   * Used to show visual feedback during drag-and-drop
   */
  const [dragActive, setDragActive] = useState(false);

  // ========================================================================
  // EVENT HANDLERS
  // ========================================================================

  /**
   * Handle file selection from input
   *
   * This runs when user selects a file using the file input
   * Event flow: User clicks input → Selects file → onChange fires
   */
  const handleFileChange = (event) => {
    // Get the selected file from the event
    // event.target is the <input> element
    // event.target.files is a FileList (like an array)
    // [0] gets the first file (we only allow one file)
    const file = event.target.files[0];

    if (file) {
      // Call our helper function to process the file
      processFile(file);
    }
  };

  /**
   * Handle drag and drop events
   *
   * These handlers enable drag-and-drop functionality
   * User can drag an image file onto the component
   */

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    // Prevent default behavior (opening file in browser)
    e.preventDefault();
    e.stopPropagation();

    setDragActive(false);

    // Get the dropped file
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  /**
   * Process a selected file
   *
   * This helper function:
   * 1. Validates the file is an image
   * 2. Creates a preview URL
   * 3. Updates the state
   *
   * @param {File} file - The file to process
   */
  const processFile = (file) => {
    // Validate file type
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file');
      return;
    }

    // Validate file size (max 10MB)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
      alert('File size must be less than 10MB');
      return;
    }

    // Clean up previous preview URL to free memory
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    // Create a temporary URL for the image preview
    // This URL only exists in the browser's memory
    const url = URL.createObjectURL(file);

    // Update state with the new file and preview
    setSelectedFile(file);
    setPreviewUrl(url);
  };

  /**
   * Handle upload button click
   *
   * This runs when user clicks "Classify Image"
   * It calls the onUpload function passed from the parent component
   */
  const handleUpload = () => {
    if (selectedFile) {
      // Call the onUpload prop function with the selected file
      // The parent component (App.js) will handle the actual upload
      onUpload(selectedFile);
    }
  };

  /**
   * Clear the selected image
   *
   * Resets the component to its initial state
   */
  const handleClear = () => {
    // Clean up the preview URL
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    // Reset state
    setSelectedFile(null);
    setPreviewUrl(null);
  };

  // ========================================================================
  // RENDER
  // ========================================================================

  return (
    <div className="image-upload">
      {/* Header */}
      <h2 className="upload-title">Upload an Image</h2>
      <p className="upload-subtitle">
        Select or drag & drop an image to classify with AI
      </p>

      {/* Drop Zone */}
      <div
        className={`drop-zone ${dragActive ? 'drag-active' : ''} ${selectedFile ? 'has-file' : ''}`}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        {/* Show preview if file is selected */}
        {previewUrl ? (
          <div className="preview-container">
            <img
              src={previewUrl}
              alt="Preview"
              className="preview-image"
            />
            <p className="file-name">{selectedFile.name}</p>
            <p className="file-info">
              {(selectedFile.size / 1024).toFixed(2)} KB
            </p>
          </div>
        ) : (
          // Show upload prompt if no file selected
          <div className="upload-prompt">
            <svg
              className="upload-icon"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
            </svg>
            <p className="prompt-text">
              Drag & drop an image here, or click to select
            </p>
            <p className="prompt-hint">
              Supports: JPG, PNG, GIF (max 10MB)
            </p>
          </div>
        )}

        {/* Hidden file input */}
        {/* Clicking the drop zone will trigger this input */}
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="file-input"
          disabled={isLoading}
        />
      </div>

      {/* Action Buttons */}
      {selectedFile && (
        <div className="button-group">
          {/* Upload Button */}
          <button
            onClick={handleUpload}
            disabled={isLoading}
            className="btn btn-primary"
          >
            {isLoading ? (
              <>
                <span className="spinner"></span>
                Classifying...
              </>
            ) : (
              'Classify Image'
            )}
          </button>

          {/* Clear Button */}
          <button
            onClick={handleClear}
            disabled={isLoading}
            className="btn btn-secondary"
          >
            Clear
          </button>
        </div>
      )}
    </div>
  );
}

/**
 * Export the component
 *
 * This allows other files to import it:
 *   import ImageUpload from './components/ImageUpload';
 */
export default ImageUpload;

/**
 * Component Lifecycle (for students):
 *
 * 1. Component Mounts:
 *    - Initial state set (null values)
 *    - Component rendered with upload prompt
 *
 * 2. User Interaction:
 *    - User selects file → handleFileChange runs
 *    - State updates → Component re-renders
 *    - Preview shown
 *
 * 3. User Clicks Upload:
 *    - handleUpload runs
 *    - onUpload prop function called
 *    - Parent component handles upload
 *
 * 4. Component Updates:
 *    - When isLoading prop changes
 *    - Button text changes ("Classify Image" → "Classifying...")
 *    - Button disabled during upload
 *
 * 5. Component Unmounts:
 *    - Clean up preview URL (prevent memory leaks)
 *
 * Common Patterns:
 * - Controlled vs Uncontrolled: This is uncontrolled (file input)
 * - State Management: useState for local state
 * - Props: Receive functions from parent
 * - Event Handling: Multiple event types (click, change, drag)
 */
