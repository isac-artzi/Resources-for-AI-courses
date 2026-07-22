"""Pydantic request/response models shared by the API and (optionally) the UI.

The contract between the three clouds. The "See-Sense" product is image
classification with a small CNN plus Grad-CAM explainability. The *shape* of the
contract is identical to the base template; only the fields change.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class DatasetCreate(BaseModel):
    """Request body for POST /datasets."""

    name: str = Field(..., min_length=1, max_length=100)
    n_rows: int = Field(800, ge=100, le=20_000)
    noise: float = Field(0.1, ge=0.0, le=1.0, description="Pixel noise std.")
    seed: int = Field(42, description="Seed for the synthetic image generator.")


class Dataset(BaseModel):
    """A dataset row as stored in Supabase."""

    id: int
    name: str
    n_rows: int
    img_size: int
    n_classes: int
    created_at: datetime


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
class TrainRequest(BaseModel):
    """Request body for POST /train."""

    dataset_id: int
    lr: float = Field(0.01, gt=0.0, description="Adam learning rate.")
    batch_size: int = Field(32, ge=1)
    epochs: int = Field(15, ge=1, le=500)
    test_size: float = Field(0.2, gt=0.0, lt=1.0, description="Held-out fraction.")


class ClassMetrics(BaseModel):
    accuracy: float
    macro_f1: float


class Run(BaseModel):
    """A training-run row as stored in Supabase (metrics only)."""

    id: int
    dataset_id: int
    lr: float
    batch_size: int
    epochs: int
    accuracy: float
    macro_f1: float
    created_at: datetime


class TrainResponse(BaseModel):
    run_id: int
    metrics: ClassMetrics


# ---------------------------------------------------------------------------
# Prediction  (Grad-CAM explainability)
# ---------------------------------------------------------------------------
class PredictResponse(BaseModel):
    run_id: int
    label: int
    class_name: str
    confidence: float = Field(..., description="Softmax probability of the top class.")
    probs: List[float]
    image_sha256: str = Field(..., description="Hash of the input image (no pixels stored).")
    input_png: str = Field(..., description="base64 PNG of the (preprocessed) input.")
    cam_png: str = Field(..., description="base64 PNG of the Grad-CAM overlay.")
    true_class: Optional[str] = Field(
        None, description="Ground-truth class when predicting a generated sample."
    )


class PredictSampleRequest(BaseModel):
    """Request body for POST /predict_sample (no file upload needed)."""

    run_id: int
    seed: int = Field(0, description="Seed for the generated sample.")
    true_class: Optional[int] = Field(
        None, ge=0, description="Force a class index; None picks one at random."
    )


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------
class ClassesResponse(BaseModel):
    classes: List[str]
    n_classes: int


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
