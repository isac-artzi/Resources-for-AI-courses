"""Pydantic request/response models shared by the API and (optionally) the UI.

Keeping every wire-format type in one module is the contract between the three
clouds. The Streamlit UI never imports model or SQL code -- it only imports (or
mirrors) these schemas so that the payloads it sends match what FastAPI expects.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class DatasetCreate(BaseModel):
    """Request body for POST /datasets."""

    name: str = Field(..., min_length=1, max_length=100)
    slope: float = Field(2.0, description="True slope of the generating line.")
    intercept: float = Field(1.0, description="True intercept of the generating line.")
    noise: float = Field(1.0, ge=0.0, description="Std-dev of the Gaussian noise.")
    n_points: int = Field(500, ge=50, le=100_000)


class Dataset(BaseModel):
    """A dataset row as stored in Supabase."""

    id: int
    name: str
    slope: float
    intercept: float
    noise: float
    n_points: int
    created_at: datetime


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
class TrainRequest(BaseModel):
    """Request body for POST /train."""

    dataset_id: int
    lr: float = Field(0.01, gt=0.0, description="Learning rate for gradient descent.")
    batch_size: int = Field(32, ge=1)
    epochs: int = Field(100, ge=1, le=5000)
    test_size: float = Field(0.2, gt=0.0, lt=1.0, description="Held-out fraction.")


class Metrics(BaseModel):
    mse: float
    mae: float
    r2: float


class Run(BaseModel):
    """A training-run row as stored in Supabase."""

    id: int
    dataset_id: int
    lr: float
    batch_size: int
    epochs: int
    mse: float
    mae: float
    r2: float
    weights_json: dict
    created_at: datetime


class TrainResponse(BaseModel):
    run_id: int
    metrics: Metrics
    weights: dict = Field(..., description="Fitted {'slope': .., 'intercept': ..}.")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """Request body for POST /predict."""

    run_id: int
    x: float = Field(..., description="Single feature value to predict on.")


class PredictResponse(BaseModel):
    run_id: int
    x: float
    yhat: float


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------
class Health(BaseModel):
    status: str
    model_loader: bool
    supabase: bool


class Version(BaseModel):
    git_sha: str
    torch_version: str
    supabase_project_ref: Optional[str]
