/**
 * TodoItem Component
 *
 * Displays a single todo item with:
 * - Checkbox to mark complete/incomplete
 * - Title and description
 * - Delete button
 * - Timestamps
 */

import { useState } from 'react';
import type { Todo } from '../types';

interface TodoItemProps {
  todo: Todo;
  onToggle: (id: number, completed: boolean) => Promise<void>;
  onDelete: (id: number) => Promise<void>;
}

export default function TodoItem({ todo, onToggle, onDelete }: TodoItemProps) {
  const [isUpdating, setIsUpdating] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  /**
   * Handle checkbox toggle
   * Updates the todo's completion status
   */
  const handleToggle = async () => {
    setIsUpdating(true);
    try {
      await onToggle(todo.id, !todo.completed);
    } catch (error) {
      console.error('Error toggling todo:', error);
    } finally {
      setIsUpdating(false);
    }
  };

  /**
   * Handle delete button click
   * Shows confirmation before deleting
   */
  const handleDelete = async () => {
    if (!window.confirm('Are you sure you want to delete this todo?')) {
      return;
    }

    setIsDeleting(true);
    try {
      await onDelete(todo.id);
    } catch (error) {
      console.error('Error deleting todo:', error);
      setIsDeleting(false); // Re-enable if error occurs
    }
  };

  /**
   * Format date for display
   * Converts ISO string to readable format
   */
  const formatDate = (dateString: string): string => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className={`todo-item ${todo.completed ? 'completed' : ''} ${isDeleting ? 'deleting' : ''}`}>
      {/* Checkbox for completion status */}
      <div className="todo-checkbox">
        <input
          type="checkbox"
          id={`todo-${todo.id}`}
          checked={todo.completed}
          onChange={handleToggle}
          disabled={isUpdating || isDeleting}
        />
        <label htmlFor={`todo-${todo.id}`} className="checkbox-label"></label>
      </div>

      {/* Todo content */}
      <div className="todo-content">
        <h3 className="todo-title">{todo.title}</h3>
        {todo.description && (
          <p className="todo-description">{todo.description}</p>
        )}
        <div className="todo-meta">
          <span className="todo-date" title={`Created: ${formatDate(todo.created_at)}`}>
            {formatDate(todo.created_at)}
          </span>
          {todo.updated_at !== todo.created_at && (
            <span className="todo-updated" title={`Updated: ${formatDate(todo.updated_at)}`}>
              (updated)
            </span>
          )}
        </div>
      </div>

      {/* Delete button */}
      <button
        className="btn btn-delete"
        onClick={handleDelete}
        disabled={isDeleting}
        aria-label="Delete todo"
        title="Delete"
      >
        {isDeleting ? '...' : '×'}
      </button>
    </div>
  );
}
