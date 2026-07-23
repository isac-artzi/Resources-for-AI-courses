/**
 * TodoList Component
 *
 * Displays a list of todos with filtering options.
 * Features:
 * - Filter by all/active/completed
 * - Todo statistics
 * - Empty state
 */

import { useState } from 'react';
import TodoItem from './TodoItem';
import type { Todo } from '../types';

interface TodoListProps {
  todos: Todo[];
  onToggle: (id: number, completed: boolean) => Promise<void>;
  onDelete: (id: number) => Promise<void>;
}

type FilterType = 'all' | 'active' | 'completed';

export default function TodoList({ todos, onToggle, onDelete }: TodoListProps) {
  const [filter, setFilter] = useState<FilterType>('all');

  /**
   * Filter todos based on selected filter
   */
  const filteredTodos = todos.filter((todo) => {
    if (filter === 'active') return !todo.completed;
    if (filter === 'completed') return todo.completed;
    return true; // 'all'
  });

  /**
   * Calculate statistics
   */
  const totalTodos = todos.length;
  const activeTodos = todos.filter((t) => !t.completed).length;
  const completedTodos = todos.filter((t) => t.completed).length;

  /**
   * Render empty state
   */
  if (todos.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-icon">📝</div>
        <h3>No todos yet</h3>
        <p>Add your first todo to get started!</p>
      </div>
    );
  }

  return (
    <div className="todo-list-container">
      {/* Filter buttons */}
      <div className="filter-buttons">
        <button
          className={`filter-btn ${filter === 'all' ? 'active' : ''}`}
          onClick={() => setFilter('all')}
        >
          All <span className="badge">{totalTodos}</span>
        </button>
        <button
          className={`filter-btn ${filter === 'active' ? 'active' : ''}`}
          onClick={() => setFilter('active')}
        >
          Active <span className="badge">{activeTodos}</span>
        </button>
        <button
          className={`filter-btn ${filter === 'completed' ? 'active' : ''}`}
          onClick={() => setFilter('completed')}
        >
          Completed <span className="badge">{completedTodos}</span>
        </button>
      </div>

      {/* Todo items */}
      <div className="todo-list">
        {filteredTodos.length === 0 ? (
          <div className="empty-filter-state">
            <p>No {filter} todos</p>
          </div>
        ) : (
          filteredTodos.map((todo) => (
            <TodoItem
              key={todo.id}
              todo={todo}
              onToggle={onToggle}
              onDelete={onDelete}
            />
          ))
        )}
      </div>

      {/* Summary */}
      {totalTodos > 0 && (
        <div className="todo-summary">
          <span>
            {activeTodos} {activeTodos === 1 ? 'task' : 'tasks'} remaining
          </span>
          {completedTodos > 0 && (
            <span className="completed-count">
              {completedTodos} completed
            </span>
          )}
        </div>
      )}
    </div>
  );
}
