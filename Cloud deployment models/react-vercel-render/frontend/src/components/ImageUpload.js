/**
 * ImageUpload Component
 *
 * Handles image file selection, preview, and upload form
 */

import React from 'react';
import './ImageUpload.css';

function ImageUpload({ preview, selectedFile, onFileSelect, onSubmit, onClear, loading }) {
  return (
    <div className="upload-section">
      <form onSubmit={onSubmit} className="upload-form">

        {/* Preview Area */}
        {preview ? (
          <div className="preview-container">
            <img src={preview} alt="Preview" className="preview-image" />
            <div className="preview-info">
              <p className="file-name">{selectedFile.name}</p>
              <p className="file-size">
                {(selectedFile.size / 1024).toFixed(2)} KB
              </p>
            </div>
          </div>
        ) : (
          <div className="upload-placeholder">
            <div className="upload-icon">📸</div>
            <p>No image selected</p>
          </div>
        )}

        {/* File Input */}
        <div className="file-input-wrapper">
          <label htmlFor="file-input" className="file-input-label">
            Choose Image
          </label>
          <input
            id="file-input"
            type="file"
            accept="image/*"
            onChange={onFileSelect}
            className="file-input"
            disabled={loading}
          />
        </div>

        {/* Action Buttons */}
        <div className="button-group">
          <button
            type="submit"
            className="btn btn-primary"
            disabled={!selectedFile || loading}
          >
            {loading ? 'Classifying...' : 'Classify Image'}
          </button>

          {selectedFile && !loading && (
            <button
              type="button"
              onClick={onClear}
              className="btn btn-secondary"
            >
              Clear
            </button>
          )}
        </div>
      </form>
    </div>
  );
}

export default ImageUpload;
