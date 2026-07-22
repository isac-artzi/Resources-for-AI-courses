"""
Data Models for Todo Application

This module defines Pydantic models for data validation and serialization.
Pydantic ensures type safety and automatic validation of incoming/outgoing data.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class TodoBase(BaseModel):
    """
    Base Todo model with common attributes.
    This is used as a base class for other Todo models.
    """
    title: str = Field(..., min_length=1, max_length=200, description="Title of the todo item")
    description: Optional[str] = Field(None, max_length=1000, description="Optional description")
    completed: bool = Field(default=False, description="Whether the todo is completed")


class TodoCreate(TodoBase):
    """
    Model for creating a new Todo.
    Inherits from TodoBase and doesn't add extra fields.
    Used for POST requests to validate incoming data.
    """
    pass


class TodoUpdate(BaseModel):
    """
    Model for updating an existing Todo.
    All fields are optional to allow partial updates (PATCH-style).
    """
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    completed: Optional[bool] = None


class Todo(TodoBase):
    """
    Complete Todo model with all fields including ID and timestamps.
    This represents the full todo item as stored and returned by the API.

    Config:
        from_attributes: Allows converting ORM models to Pydantic models
        This is useful when upgrading to use a real database with SQLAlchemy
    """
    id: int = Field(..., description="Unique identifier for the todo")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")

    class Config:
        from_attributes = True  # Allows ORM mode for future database integration


class HealthCheck(BaseModel):
    """
    Model for health check endpoint response.
    Used to verify the API is running correctly.
    """
    status: str = Field(..., description="Health status")
    message: str = Field(..., description="Health message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
