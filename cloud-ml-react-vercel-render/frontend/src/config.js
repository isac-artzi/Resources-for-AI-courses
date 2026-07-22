/**
 * Configuration file for API endpoints
 *
 * This file manages API URLs for different environments.
 * Update REACT_APP_API_URL in .env for production deployment.
 */

// Get API URL from environment variable or use default
export const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// API endpoints
export const ENDPOINTS = {
  root: `${API_URL}/`,
  predict: `${API_URL}/predict`,
  predictBatch: `${API_URL}/predict/batch`,
  health: `${API_URL}/health`,
  modelInfo: `${API_URL}/model/info`,
};

// Configuration
export const CONFIG = {
  maxFileSize: 10 * 1024 * 1024, // 10MB
  allowedFileTypes: ['image/jpeg', 'image/png', 'image/jpg', 'image/webp'],
  maxBatchSize: 10,
};

export default {
  API_URL,
  ENDPOINTS,
  CONFIG,
};
