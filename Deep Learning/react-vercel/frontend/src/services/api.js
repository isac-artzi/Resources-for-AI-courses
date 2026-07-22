/**
 * API Service Module
 * ==================
 *
 * This module handles all communication with the FastAPI backend.
 * It provides a clean interface for making API requests.
 *
 * Features:
 * - Centralized API configuration
 * - Environment-based URL configuration
 * - Error handling and logging
 * - Type-safe API calls
 *
 * Author: Cloud ML Deployment
 */

import axios from 'axios'

// ============================================================================
// Configuration
// ============================================================================

/**
 * Get API base URL from environment variables
 *
 * Environment variables in Vite:
 * - Must start with VITE_ to be exposed to the app
 * - Defined in .env.local for development
 * - Defined in Vercel for production
 *
 * Fallback to localhost for development if not set
 */
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Request timeout (30 seconds)
// Adjust based on model inference time
const TIMEOUT = 30000

console.log('API Configuration:', { API_BASE_URL, TIMEOUT })

// ============================================================================
// Axios Instance
// ============================================================================

/**
 * Create an axios instance with default configuration
 * This instance is used for all API requests
 */
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: TIMEOUT,
  headers: {
    'Content-Type': 'multipart/form-data',
  }
})

// ============================================================================
// Request Interceptor
// ============================================================================

/**
 * Add request interceptor for logging and debugging
 * Runs before every request is sent
 */
apiClient.interceptors.request.use(
  (config) => {
    console.log('🚀 API Request:', {
      method: config.method,
      url: config.url,
      baseURL: config.baseURL
    })
    return config
  },
  (error) => {
    console.error('❌ Request Error:', error)
    return Promise.reject(error)
  }
)

// ============================================================================
// Response Interceptor
// ============================================================================

/**
 * Add response interceptor for logging and error handling
 * Runs after every response is received
 */
apiClient.interceptors.response.use(
  (response) => {
    console.log('✅ API Response:', {
      status: response.status,
      data: response.data
    })
    return response
  },
  (error) => {
    console.error('❌ Response Error:', {
      message: error.message,
      response: error.response?.data,
      status: error.response?.status
    })

    // Enhance error message for better user feedback
    if (error.code === 'ECONNREFUSED' || error.code === 'ERR_NETWORK') {
      error.message = 'Cannot connect to server. Is the backend running?'
    } else if (error.code === 'ECONNABORTED') {
      error.message = 'Request timeout. The server took too long to respond.'
    } else if (error.response) {
      // Server responded with error status
      error.message = error.response.data.detail || error.response.data.message || error.message
    }

    return Promise.reject(error)
  }
)

// ============================================================================
// API Functions
// ============================================================================

/**
 * Classify an image using the ML model
 *
 * This function:
 * 1. Creates a FormData object with the image file
 * 2. Sends POST request to /predict endpoint
 * 3. Returns the prediction results
 *
 * @param {File} imageFile - The image file to classify
 * @returns {Promise<Object>} - Prediction results
 *
 * @example
 * const file = // ... get file from input
 * const results = await classifyImage(file)
 * console.log(results.predictions)
 *
 * Expected Response:
 * {
 *   success: true,
 *   predictions: [
 *     { class: "Golden Retriever", confidence: 0.89, ... },
 *     { class: "Labrador Retriever", confidence: 0.06, ... }
 *   ],
 *   processing_time: 0.234,
 *   model: "MobileNetV2"
 * }
 */
export async function classifyImage(imageFile) {
  try {
    // Validate input
    if (!imageFile) {
      throw new Error('No image file provided')
    }

    if (!(imageFile instanceof File)) {
      throw new Error('Invalid file object')
    }

    console.log('Classifying image:', {
      name: imageFile.name,
      type: imageFile.type,
      size: imageFile.size
    })

    // ========================================================================
    // Create FormData
    // ========================================================================
    // FormData is used to send files in HTTP requests
    // It creates multipart/form-data format expected by FastAPI
    const formData = new FormData()
    formData.append('file', imageFile)

    // Log FormData contents (for debugging)
    console.log('FormData contents:')
    for (let [key, value] of formData.entries()) {
      console.log(`  ${key}:`, value instanceof File ? value.name : value)
    }

    // ========================================================================
    // Send Request
    // ========================================================================
    const response = await apiClient.post('/predict', formData)

    // ========================================================================
    // Validate Response
    // ========================================================================
    if (!response.data) {
      throw new Error('Empty response from server')
    }

    if (!response.data.success) {
      throw new Error(response.data.error || 'Classification failed')
    }

    if (!response.data.predictions || response.data.predictions.length === 0) {
      throw new Error('No predictions returned')
    }

    // ========================================================================
    // Return Results
    // ========================================================================
    return response.data

  } catch (error) {
    console.error('Classification failed:', error)
    throw error
  }
}

/**
 * Check API health status
 * Useful for testing if the backend is running
 *
 * @returns {Promise<Object>} - Health status
 *
 * @example
 * const health = await checkHealth()
 * console.log(health.status) // "healthy"
 */
export async function checkHealth() {
  try {
    const response = await apiClient.get('/health')
    return response.data
  } catch (error) {
    console.error('Health check failed:', error)
    throw error
  }
}

/**
 * Get API root information
 * Basic endpoint to verify API is responding
 *
 * @returns {Promise<Object>} - API info
 */
export async function getApiInfo() {
  try {
    const response = await apiClient.get('/')
    return response.data
  } catch (error) {
    console.error('API info request failed:', error)
    throw error
  }
}

// ============================================================================
// Export API client for advanced use
// ============================================================================
export default apiClient

/**
 * Learning Notes:
 * ==============
 *
 * 1. Axios:
 *    - Popular HTTP client library
 *    - Promise-based API
 *    - Interceptors for request/response
 *    - Automatic JSON transformation
 *
 * 2. Environment Variables:
 *    - import.meta.env in Vite
 *    - process.env in Node.js
 *    - Must start with VITE_ in Vite
 *    - Different values for dev/production
 *
 * 3. FormData:
 *    - Built-in browser API
 *    - Used for file uploads
 *    - Creates multipart/form-data
 *    - Works with FastAPI's UploadFile
 *
 * 4. Async/Await:
 *    - async function returns Promise
 *    - await pauses until Promise resolves
 *    - try/catch for error handling
 *    - Makes async code look synchronous
 *
 * 5. Interceptors:
 *    - Run before/after requests
 *    - Useful for logging, auth, error handling
 *    - Can modify requests/responses
 *    - Global configuration
 *
 * 6. Error Handling:
 *    - Network errors (no connection)
 *    - Timeout errors
 *    - HTTP errors (4xx, 5xx)
 *    - Validation errors
 *    - Always provide user-friendly messages
 *
 * 7. Module Exports:
 *    - export function - named export
 *    - export default - default export
 *    - Import: import { func } from './module'
 *    - Import default: import name from './module'
 */
