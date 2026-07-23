"""
API Routes for Todo Application

This module defines all the API endpoints (routes) for the Todo application.
Each route handles a specific HTTP method and path.

Endpoints:
- GET    /              - Health check
- GET    /api/todos     - Get all todos
- POST   /api/todos     - Create a new todo
- GET    /api/todos/{id} - Get a specific todo
- PUT    /api/todos/{id} - Update a todo (full update)
- DELETE /api/todos/{id} - Delete a todo
"""

from fastapi import APIRouter, HTTPException, status
from typing import List
from datetime import datetime
from .models import Todo, TodoCreate, TodoUpdate, HealthCheck

# Create an API router instance
# This organizes routes and can be included in the main app
router = APIRouter()

# In-memory storage for todos
# NOTE: This resets when the server restarts
# For production, use a database like PostgreSQL or MongoDB
todos_db: List[Todo] = []

# Counter for generating unique IDs
# In a real database, this would be handled automatically
next_id: int = 1


@router.get("/", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify the API is running.

    Returns:
        HealthCheck: Status information about the API

    Example Response:
        {
            "status": "healthy",
            "message": "Todo API is running",
            "timestamp": "2024-01-01T12:00:00"
        }
    """
    return HealthCheck(
        status="healthy",
        message="Todo API is running"
    )


@router.get("/api/todos", response_model=List[Todo], tags=["Todos"])
async def get_todos():
    """
    Retrieve all todos from the database.

    Returns:
        List[Todo]: List of all todo items

    Example Response:
        [
            {
                "id": 1,
                "title": "Buy groceries",
                "description": "Milk, eggs, bread",
                "completed": false,
                "created_at": "2024-01-01T12:00:00",
                "updated_at": "2024-01-01T12:00:00"
            }
        ]
    """
    return todos_db


@router.post("/api/todos", response_model=Todo, status_code=status.HTTP_201_CREATED, tags=["Todos"])
async def create_todo(todo: TodoCreate):
    """
    Create a new todo item.

    Args:
        todo (TodoCreate): Todo data from request body

    Returns:
        Todo: The newly created todo with ID and timestamps

    Example Request Body:
        {
            "title": "Buy groceries",
            "description": "Milk, eggs, bread",
            "completed": false
        }
    """
    global next_id

    # Create a new Todo instance with all fields
    new_todo = Todo(
        id=next_id,
        title=todo.title,
        description=todo.description,
        completed=todo.completed,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    # Add to database and increment ID counter
    todos_db.append(new_todo)
    next_id += 1

    return new_todo


@router.get("/api/todos/{todo_id}", response_model=Todo, tags=["Todos"])
async def get_todo(todo_id: int):
    """
    Retrieve a specific todo by ID.

    Args:
        todo_id (int): The ID of the todo to retrieve

    Returns:
        Todo: The requested todo item

    Raises:
        HTTPException: 404 if todo not found
    """
    # Search for the todo in the database
    for todo in todos_db:
        if todo.id == todo_id:
            return todo

    # If not found, raise 404 error
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Todo with id {todo_id} not found"
    )


@router.put("/api/todos/{todo_id}", response_model=Todo, tags=["Todos"])
async def update_todo(todo_id: int, todo_update: TodoUpdate):
    """
    Update an existing todo item.
    Only provided fields will be updated (partial update).

    Args:
        todo_id (int): The ID of the todo to update
        todo_update (TodoUpdate): Fields to update

    Returns:
        Todo: The updated todo item

    Raises:
        HTTPException: 404 if todo not found

    Example Request Body:
        {
            "completed": true
        }
    """
    # Find the todo to update
    for todo in todos_db:
        if todo.id == todo_id:
            # Update only the fields that were provided
            # exclude_unset=True means only update fields that were explicitly set
            update_data = todo_update.model_dump(exclude_unset=True)

            for field, value in update_data.items():
                setattr(todo, field, value)

            # Update the timestamp
            todo.updated_at = datetime.utcnow()

            return todo

    # If not found, raise 404 error
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Todo with id {todo_id} not found"
    )


@router.delete("/api/todos/{todo_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Todos"])
async def delete_todo(todo_id: int):
    """
    Delete a todo item.

    Args:
        todo_id (int): The ID of the todo to delete

    Returns:
        None (204 No Content status)

    Raises:
        HTTPException: 404 if todo not found
    """
    global todos_db

    # Find and remove the todo
    for index, todo in enumerate(todos_db):
        if todo.id == todo_id:
            todos_db.pop(index)
            return  # Return 204 No Content

    # If not found, raise 404 error
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Todo with id {todo_id} not found"
    )
