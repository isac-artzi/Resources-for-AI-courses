/**
 * API Service - Communication with Backend
 *
 * This file handles all communication with the FastAPI backend.
 * It centralizes API calls so we don't repeat code in components.
 *
 * Key Concepts:
 * - API: Application Programming Interface (how frontend talks to backend)
 * - HTTP Methods: GET (read), POST (create), PUT (update), DELETE (remove)
 * - FormData: Special format for sending files
 * - Async/Await: Handle asynchronous operations
 */

/**
 * Configuration
 *
 * API_URL: The base URL of our backend
 * - During development: http://localhost:8000 (backend on your machine)
 * - After deployment: https://your-backend.onrender.com (backend on Render)
 *
 * IMPORTANT: Update this URL when deploying!
 */

// Check if we're in production or development
// process.env.NODE_ENV is set by React automatically
const isDevelopment = process.env.NODE_ENV === 'development';

// Use environment variable if available, otherwise use localhost
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Log the API URL for debugging
console.log(`API URL: ${API_URL} (${isDevelopment ? 'Development' : 'Production'})`);

/**
 * API Service Object
 *
 * This object contains all our API methods
 * Benefits of this approach:
 * - All API logic in one place
 * - Easy to maintain and update
 * - Can add error handling, retries, etc.
 */
const api = {

  /**
   * Check if the backend is healthy
   *
   * Purpose: Verify the backend is running and responding
   * Method: GET
   * Endpoint: /health
   *
   * Returns:
   *   - Success: { status: 'healthy', model_loaded: true }
   *   - Error: null (if backend is down)
   *
   * Example:
   *   const health = await api.checkHealth();
   *   if (health) {
   *     console.log('Backend is running!');
   *   }
   */
  checkHealth: async () => {
    try {
      // Make a GET request to the health endpoint
      const response = await fetch(`${API_URL}/health`);

      // Check if the request was successful (status code 200-299)
      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
      }

      // Parse the JSON response
      const data = await response.json();

      return data;

    } catch (error) {
      // If anything goes wrong (network error, backend down, etc.)
      console.error('Health check error:', error);
      return null;
    }
  },

  /**
   * Upload and classify an image
   *
   * Purpose: Send an image to the backend and get predictions
   * Method: POST
   * Endpoint: /predict
   *
   * Parameters:
   *   imageFile: File object from <input type="file">
   *
   * Returns:
   *   - Success: {
   *       success: true,
   *       filename: "cat.jpg",
   *       predictions: [
   *         { label: "tabby cat", confidence: 0.92, class_id: 281 },
   *         ...
   *       ]
   *     }
   *   - Error: { error: "error message" }
   *
   * Example:
   *   const result = await api.uploadImage(imageFile);
   *   if (result.success) {
   *     console.log('Top prediction:', result.predictions[0]);
   *   }
   */
  uploadImage: async (imageFile) => {
    try {
      // ----------------------------------------------------------------
      // STEP 1: Validate the input
      // ----------------------------------------------------------------

      if (!imageFile) {
        throw new Error('No image file provided');
      }

      // Check if it's actually a file
      if (!(imageFile instanceof File)) {
        throw new Error('Invalid file object');
      }

      // Check file type (basic validation)
      if (!imageFile.type.startsWith('image/')) {
        throw new Error('File must be an image');
      }

      // Check file size (limit to 10MB)
      const maxSize = 10 * 1024 * 1024; // 10MB in bytes
      if (imageFile.size > maxSize) {
        throw new Error('File size must be less than 10MB');
      }

      // ----------------------------------------------------------------
      // STEP 2: Prepare the request
      // ----------------------------------------------------------------

      // Create a FormData object
      // FormData is a special format for sending files in HTTP requests
      // It's like a form you'd submit on a website
      const formData = new FormData();

      // Add the image file to the form data
      // 'file' is the key name the backend expects (see backend/app.py)
      formData.append('file', imageFile);

      // Log for debugging
      console.log('Uploading image:', {
        name: imageFile.name,
        size: `${(imageFile.size / 1024).toFixed(2)} KB`,
        type: imageFile.type
      });

      // ----------------------------------------------------------------
      // STEP 3: Make the API request
      // ----------------------------------------------------------------

      // Send POST request to the predict endpoint
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',

        // Send the FormData as the request body
        body: formData,

        // Note: We DON'T set Content-Type header
        // The browser automatically sets it to 'multipart/form-data'
        // with the correct boundary for FormData
      });

      // ----------------------------------------------------------------
      // STEP 4: Handle the response
      // ----------------------------------------------------------------

      // Parse the JSON response
      const data = await response.json();

      // Check if the request was successful
      if (!response.ok) {
        // Backend returned an error (4xx or 5xx status code)
        throw new Error(data.detail || 'Prediction failed');
      }

      // Log successful prediction
      console.log('Prediction successful:', data);

      return data;

    } catch (error) {
      // ----------------------------------------------------------------
      // STEP 5: Error handling
      // ----------------------------------------------------------------

      console.error('Upload error:', error);

      // Return error in a consistent format
      return {
        success: false,
        error: error.message || 'Failed to upload image'
      };
    }
  },

  /**
   * Get model information
   *
   * Purpose: Get details about the loaded model
   * Method: GET
   * Endpoint: /model-info
   *
   * Returns:
   *   Model information object
   *
   * Example:
   *   const info = await api.getModelInfo();
   *   console.log('Model:', info.model_name);
   */
  getModelInfo: async () => {
    try {
      const response = await fetch(`${API_URL}/model-info`);

      if (!response.ok) {
        throw new Error('Failed to get model info');
      }

      const data = await response.json();
      return data;

    } catch (error) {
      console.error('Model info error:', error);
      return null;
    }
  }

};

/**
 * Export the API service
 *
 * This allows other files to import and use it:
 *   import api from './services/api';
 *   const result = await api.uploadImage(file);
 */
export default api;

/**
 * Alternative API URL Configuration for Deployment
 *
 * When deploying, you have several options:
 *
 * 1. Environment Variable (Recommended):
 *    - Create a .env file in the frontend folder:
 *      REACT_APP_API_URL=https://your-backend.onrender.com
 *    - React will automatically load variables starting with REACT_APP_
 *
 * 2. Build-time Configuration:
 *    - Set environment variable during build:
 *      REACT_APP_API_URL=https://your-backend.onrender.com npm run build
 *
 * 3. Render Environment Variable:
 *    - In Render dashboard, go to your static site
 *    - Add environment variable:
 *      Key: REACT_APP_API_URL
 *      Value: https://your-backend.onrender.com
 *
 * 4. Hard-code (Not recommended for learning):
 *    const API_URL = 'https://your-backend.onrender.com';
 *
 * Why environment variables?
 * - Different URLs for development and production
 * - Easy to change without modifying code
 * - Keep sensitive information out of code
 */
