"""Pydantic request/response models shared by the API and (optionally) the UI.

Keeping every wire-format type in one module is the contract between the three
clouds. The Streamlit UI never imports model or SQL code -- it only imports (or
mirrors) these schemas so that the payloads it sends match what FastAPI expects.

This is the Topic 2 product "Income-Insight" (tabular binary classification).
The *shape* of the contract is identical to Topic 1; only the fields change:
regression metrics -> classification metrics, a scalar feature -> a record.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class DatasetCreate(BaseModel):
    """Request body for POST /datasets."""

    name: str = Field(..., min_length=1, max_length=100)
    n_rows: int = Field(2000, ge=200, le=100_000)
    noise: float = Field(1.0, ge=0.0, description="Scales the label-flip fraction.")
    seed: int = Field(42, description="Seed for the synthetic generator.")


class Dataset(BaseModel):
    """A dataset row as stored in Supabase."""

    id: int
    name: str
    n_rows: int
    n_features: int
    positive_rate: float
    created_at: datetime


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
class TrainRequest(BaseModel):
    """Request body for POST /train."""

    dataset_id: int
    hidden_dim: int = Field(32, ge=1, le=1024, description="MLP hidden-layer width.")
    lr: float = Field(0.01, gt=0.0, description="Adam learning rate.")
    batch_size: int = Field(32, ge=1)
    epochs: int = Field(100, ge=1, le=5000)
    test_size: float = Field(0.2, gt=0.0, lt=1.0, description="Held-out fraction.")


class ClassMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


class Run(BaseModel):
    """A training-run row as stored in Supabase (metrics only; no model blob)."""

    id: int
    dataset_id: int
    hidden_dim: int
    lr: float
    batch_size: int
    epochs: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    created_at: datetime


class TrainResponse(BaseModel):
    run_id: int
    metrics: ClassMetrics


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """Request body for POST /predict -- one record keyed by FEATURE_COLS."""

    run_id: int
    features: Dict[str, object] = Field(
        ..., description="One record, e.g. {'age': 39, 'education_num': 13, ...}."
    )


class PredictResponse(BaseModel):
    run_id: int
    label: int = Field(..., description="0 = <=50K, 1 = >50K.")
    income: str = Field(..., description="Human-readable class label.")
    proba: float = Field(..., description="P(income > 50K).")


class PredictBatchRequest(BaseModel):
    """Request body for POST /predict_batch."""

    run_id: int
    records: List[Dict[str, object]]


class BatchItem(BaseModel):
    label: int
    income: str
    proba: float


class PredictBatchResponse(BaseModel):
    run_id: int
    predictions: List[BatchItem]


# ---------------------------------------------------------------------------
# Schema & audit
# ---------------------------------------------------------------------------
class SchemaResponse(BaseModel):
    """Feature contract, so the UI can build its input form dynamically."""

    numeric_features: List[str]
    categorical_features: List[str]
    categories: Dict[str, List[str]]
    target_name: str
    target_classes: List[str]


class AuditGroup(BaseModel):
    group: str
    n: int
    positive_rate: float = Field(..., description="Share predicted >50K in this group.")


class AuditResponse(BaseModel):
    run_id: int
    by: str
    total: int
    overall_positive_rate: float
    groups: List[AuditGroup]


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
    sklearn_version: str
    supabase_project_ref: Optional[str]
