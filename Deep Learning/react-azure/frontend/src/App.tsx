/**
 * Main App Component
 *
 * This is the root component that orchestrates the entire application.
 * Responsibilities:
 * - Fetch todos from API on mount
 * - Manage todo state
 * - Handle CRUD operations
 * - Display loading and error states
 */

import { useEffect, useState } from 'react';
import TodoForm from './components/TodoForm';
import TodoList from './components/TodoList';
import api from './services/api';
import type { Todo, TodoCreate } from './types';

function App() {
  // State management
  const [todos, setTodos] = useState<Todo[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  /**
   * Fetch todos from API on component mount
   * useEffect with empty dependency array [] runs once on mount
   */
  useEffect(() => {
    loadTodos();
  }, []);

  /**
   * Load all todos from the API
   */
  const loadTodos = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const data = await api.getTodos();
      setTodos(data);
    } catch (err) {
      console.error('Error loading todos:', err);
      setError('Failed to load todos. Please check if the backend is running.');
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Create a new todo
   * @param todoData - Data for the new todo
   */
  const handleCreateTodo = async (todoData: TodoCreate) => {
    try {
      const newTodo = await api.createTodo(todoData);

      // Add new todo to the beginning of the list
      setTodos([newTodo, ...todos]);
    } catch (err) {
      console.error('Error creating todo:', err);
      throw err; // Re-throw to let TodoForm handle the error
    }
  };

  /**
   * Toggle todo completion status
   * @param id - Todo ID
   * @param completed - New completion status
   */
  const handleToggleTodo = async (id: number, completed: boolean) => {
    try {
      const updatedTodo = await api.toggleTodo(id, completed);

      // Update todo in state
      setTodos(todos.map((todo) =>
        todo.id === id ? updatedTodo : todo
      ));
    } catch (err) {
      console.error('Error toggling todo:', err);
      throw err;
    }
  };

  /**
   * Delete a todo
   * @param id - Todo ID
   */
  const handleDeleteTodo = async (id: number) => {
    try {
      await api.deleteTodo(id);

      // Remove todo from state
      setTodos(todos.filter((todo) => todo.id !== id));
    } catch (err) {
      console.error('Error deleting todo:', err);
      throw err;
    }
  };

  /**
   * Render loading state
   */
  if (isLoading) {
    return (
      <div className="app">
        <div className="container">
          <div className="loading">
            <div className="spinner"></div>
            <p>Loading todos...</p>
          </div>
        </div>
      </div>
    );
  }

  /**
   * Render error state
   */
  if (error) {
    return (
      <div className="app">
        <div className="container">
          <div className="error-container">
            <div className="error-icon">⚠️</div>
            <h2>Oops! Something went wrong</h2>
            <p className="error-message">{error}</p>
            <button className="btn btn-primary" onClick={loadTodos}>
              Try Again
            </button>
            <div className="error-help">
              <p><strong>Troubleshooting tips:</strong></p>
              <ul>
                <li>Make sure the backend server is running on port 8000</li>
                <li>Check the browser console for detailed errors</li>
                <li>Verify the API_URL in your .env file</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    );
  }

  /**
   * Main app render
   */
  return (
    <div className="app">
      <div className="container">
        {/* Header */}
        <header className="app-header">
          <h1 className="app-title">📝 My Todos</h1>
          <p className="app-subtitle">
            Built with React + FastAPI on Azure
          </p>
        </header>

        {/* Todo creation form */}
        <div className="section">
          <TodoForm onSubmit={handleCreateTodo} />
        </div>

        {/* Todo list */}
        <div className="section">
          <TodoList
            todos={todos}
            onToggle={handleToggleTodo}
            onDelete={handleDeleteTodo}
          />
        </div>

        {/* Footer */}
        <footer className="app-footer">
          <p>
            <a
              href="https://github.com/yourusername/react-fastapi-azure-todo"
              target="_blank"
              rel="noopener noreferrer"
            >
              View on GitHub
            </a>
          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;
