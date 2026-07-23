/**
 * TodoForm Component
 *
 * Form for creating new todo items.
 * Features:
 * - Controlled input fields
 * - Form validation
 * - Loading state during submission
 * - Error handling
 */

import { useState, FormEvent } from 'react';
import type { TodoCreate } from '../types';

interface TodoFormProps {
  onSubmit: (todo: TodoCreate) => Promise<void>;
}

export default function TodoForm({ onSubmit }: TodoFormProps) {
  // Form state
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /**
   * Handle form submission
   * Validates input, calls onSubmit prop, and resets form
   */
  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault(); // Prevent page reload
    setError(null);

    // Validate title
    if (!title.trim()) {
      setError('Title is required');
      return;
    }

    if (title.length > 200) {
      setError('Title must be less than 200 characters');
      return;
    }

    setIsSubmitting(true);

    try {
      // Create todo object
      const newTodo: TodoCreate = {
        title: title.trim(),
        description: description.trim() || undefined,
        completed: false,
      };

      // Call parent's onSubmit handler
      await onSubmit(newTodo);

      // Reset form on success
      setTitle('');
      setDescription('');
    } catch (err) {
      console.error('Error creating todo:', err);
      setError('Failed to create todo. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="todo-form">
      <div className="form-group">
        <input
          type="text"
          className="form-input"
          placeholder="What needs to be done?"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          disabled={isSubmitting}
          maxLength={200}
          autoFocus
        />
      </div>

      <div className="form-group">
        <textarea
          className="form-textarea"
          placeholder="Description (optional)"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          disabled={isSubmitting}
          maxLength={1000}
          rows={3}
        />
      </div>

      {/* Error message */}
      {error && (
        <div className="error-message" role="alert">
          {error}
        </div>
      )}

      {/* Submit button */}
      <button
        type="submit"
        className="btn btn-primary"
        disabled={isSubmitting || !title.trim()}
      >
        {isSubmitting ? 'Adding...' : 'Add Todo'}
      </button>
    </form>
  );
}
