/**
 * DrawingCanvas.js - Interactive Drawing Component
 *
 * This component provides a canvas where users can draw digits with their mouse or touch.
 * It handles:
 * - Mouse and touch events for drawing
 * - Converting drawings to image data
 * - Clearing the canvas
 *
 * React Concepts Used:
 * - useRef: To access the canvas DOM element directly
 * - useEffect: To set up event listeners when component mounts
 * - State management from parent component
 */

import React, { useRef, useEffect, useState } from 'react';

/**
 * DrawingCanvas Component
 *
 * @param {Function} onCanvasChange - Callback when the canvas is drawn on
 * @returns {JSX.Element}
 */
function DrawingCanvas({ onCanvasChange }) {
  // useRef creates a reference to the canvas DOM element
  // This allows us to access the canvas directly (like document.getElementById)
  const canvasRef = useRef(null);

  // State to track if user is currently drawing
  const [isDrawing, setIsDrawing] = useState(false);

  /**
   * useEffect runs after the component renders
   * We use it to set up the canvas and event listeners
   */
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');

    // Configure drawing style
    ctx.strokeStyle = 'black';     // Draw in black
    ctx.lineWidth = 15;            // Thick lines (easier to recognize)
    ctx.lineCap = 'round';         // Round line endings (looks better)
    ctx.lineJoin = 'round';        // Round line joints (looks better)

    // Fill canvas with white background
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, []); // Empty dependency array = run once when component mounts

  /**
   * Starts drawing when mouse/touch is pressed down
   */
  const startDrawing = (e) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Get mouse/touch position relative to canvas
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX || e.touches[0].clientX) - rect.left;
    const y = (e.clientY || e.touches[0].clientY) - rect.top;

    // Begin a new path (line)
    ctx.beginPath();
    ctx.moveTo(x, y);

    setIsDrawing(true);
  };

  /**
   * Draws as mouse/touch moves across canvas
   */
  const draw = (e) => {
    if (!isDrawing) return; // Only draw if mouse is pressed

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Get current mouse/touch position
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX || e.touches[0].clientX) - rect.left;
    const y = (e.clientY || e.touches[0].clientY) - rect.top;

    // Draw line to current position
    ctx.lineTo(x, y);
    ctx.stroke();

    // Notify parent component that canvas changed
    // This will trigger a new prediction
    if (onCanvasChange) {
      onCanvasChange(canvas);
    }
  };

  /**
   * Stops drawing when mouse/touch is released
   */
  const stopDrawing = () => {
    if (isDrawing) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      ctx.closePath(); // Finish the current path
      setIsDrawing(false);
    }
  };

  /**
   * Clears the canvas (white background)
   */
  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Fill with white
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Notify parent that canvas is now empty
    if (onCanvasChange) {
      onCanvasChange(canvas);
    }
  };

  return (
    <div className="canvas-container">
      {/* Instructions for users */}
      <p style={{ textAlign: 'center', color: '#718096', marginBottom: '1rem' }}>
        Draw a digit (0-9) with your mouse or finger
      </p>

      {/*
        The canvas element where drawing happens
        Event listeners:
        - onMouseDown/onTouchStart: Start drawing
        - onMouseMove/onTouchMove: Continue drawing
        - onMouseUp/onTouchEnd: Stop drawing
        - onMouseLeave: Stop drawing if mouse leaves canvas
      */}
      <canvas
        ref={canvasRef}
        className="drawing-canvas"
        width={280}
        height={280}
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={stopDrawing}
        onMouseLeave={stopDrawing}
        onTouchStart={startDrawing}
        onTouchMove={draw}
        onTouchEnd={stopDrawing}
      />

      {/* Controls */}
      <div className="canvas-controls">
        <button
          className="btn btn-secondary"
          onClick={clearCanvas}
        >
          Clear Canvas
        </button>
      </div>

      {/* Educational note */}
      <div className="info-message">
        <strong>How it works:</strong> Your drawing is converted to a 28x28 pixel
        image (same size as MNIST training data), then fed to the neural network
        which predicts what digit you drew!
      </div>
    </div>
  );
}

export default DrawingCanvas;
