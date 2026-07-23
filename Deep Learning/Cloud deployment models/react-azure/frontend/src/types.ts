/**
 * TypeScript Type Definitions
 *
 * This file contains all the TypeScript interfaces and types
 * used throughout the application.
 */

/**
 * Todo item structure
 * Matches the backend Todo model
 */
export interface Todo {
  id: number;
  title: string;
  description: string | null;
  completed: boolean;
  created_at: string;
  updated_at: string;
}

/**
 * Data structure for creating a new todo
 * Used when sending POST requests to the API
 */
export interface TodoCreate {
  title: string;
  description?: string;
  completed?: boolean;
}

/**
 * Data structure for updating an existing todo
 * All fields are optional for partial updates
 */
export interface TodoUpdate {
  title?: string;
  description?: string;
  completed?: boolean;
}
