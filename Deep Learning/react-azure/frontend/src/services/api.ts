/**
 * API Service Layer
 *
 * This module handles all communication with the backend API.
 * It uses axios for HTTP requests and provides a clean interface
 * for components to interact with the backend.
 */

import axios from 'axios';
import type { Todo, TodoCreate, TodoUpdate } from '../types';

// Get API URL from environment variables
// Falls back to localhost if not set
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

console.log('API URL:', API_URL); // Helpful for debugging

/**
 * Create an axios instance with default configuration
 * This allows us to set common headers, timeouts, etc.
 */
const apiClient = axios.create({
  baseURL: API_URL,
  timeout: 10000, // 10 second timeout
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Request interceptor - runs before every request
 * Useful for adding auth tokens, logging, etc.
 */
apiClient.interceptors.request.use(
  (config) => {
    console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('[API] Request error:', error);
    return Promise.reject(error);
  }
);

/**
 * Response interceptor - runs after every response
 * Useful for handling errors globally
 */
apiClient.interceptors.response.use(
  (response) => {
    console.log(`[API] Response:`, response.status, response.data);
    return response;
  },
  (error) => {
    console.error('[API] Response error:', error.response?.data || error.message);

    // You can handle specific error codes here
    if (error.response?.status === 401) {
      // Handle unauthorized (e.g., redirect to login)
      console.error('Unauthorized access');
    } else if (error.response?.status === 404) {
      console.error('Resource not found');
    } else if (error.response?.status === 500) {
      console.error('Server error');
    }

    return Promise.reject(error);
  }
);

/**
 * API Service Object
 * Contains all API methods for interacting with the backend
 */
export const api = {
  /**
   * Get all todos
   * @returns Promise<Todo[]> - Array of all todos
   */
  getTodos: async (): Promise<Todo[]> => {
    const response = await apiClient.get<Todo[]>('/api/todos');
    return response.data;
  },

  /**
   * Get a single todo by ID
   * @param id - Todo ID
   * @returns Promise<Todo> - The requested todo
   */
  getTodo: async (id: number): Promise<Todo> => {
    const response = await apiClient.get<Todo>(`/api/todos/${id}`);
    return response.data;
  },

  /**
   * Create a new todo
   * @param todo - Todo data (title, description, completed)
   * @returns Promise<Todo> - The created todo with ID
   */
  createTodo: async (todo: TodoCreate): Promise<Todo> => {
    const response = await apiClient.post<Todo>('/api/todos', todo);
    return response.data;
  },

  /**
   * Update an existing todo
   * @param id - Todo ID
   * @param updates - Fields to update
   * @returns Promise<Todo> - The updated todo
   */
  updateTodo: async (id: number, updates: TodoUpdate): Promise<Todo> => {
    const response = await apiClient.put<Todo>(`/api/todos/${id}`, updates);
    return response.data;
  },

  /**
   * Delete a todo
   * @param id - Todo ID
   * @returns Promise<void>
   */
  deleteTodo: async (id: number): Promise<void> => {
    await apiClient.delete(`/api/todos/${id}`);
  },

  /**
   * Toggle todo completion status
   * Convenience method for marking todos as complete/incomplete
   * @param id - Todo ID
   * @param completed - New completion status
   * @returns Promise<Todo> - The updated todo
   */
  toggleTodo: async (id: number, completed: boolean): Promise<Todo> => {
    return api.updateTodo(id, { completed });
  },
};

export default api;
