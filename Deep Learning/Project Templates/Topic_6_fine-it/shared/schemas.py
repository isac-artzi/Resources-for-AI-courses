"""Pydantic request/response models shared by the API and (optionally) the UI.

The contract between the three clouds. "Fine-It" is transfer learning with a
tiny causal **character Transformer**: pretrain a next-character language model
on a synthetic-dialect corpus, then fine-tune a classification head and compare
it against a from-scratch baseline. The *shape* of the contract follows the base
template; the fields add a `run_type` (pretrain vs finetune) and a generation
endpoint.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class DatasetCreate(BaseModel):
    """Request body for POST /datasets (the pretraining corpus parameters)."""

    name: str = Field(..., min_length=1, max_length=100)
    n_rows: int = Field(4000, ge=200, le=50_000, description="Corpus size for pretraining.")
    seq_len: int = Field(20, ge=6, le=64, description="Characters per string.")
    seed: int = Field(42, description="Seed for the synthetic corpus generator.")


class Dataset(BaseModel):
    """A dataset row as stored in Supabase."""

    id: int
    name: str
    n_rows: int
    seq_len: int
    vocab_size: int
    n_classes: int
    created_at: datetime


# ---------------------------------------------------------------------------
# Pretraining  (self-supervised next-char LM)
# ---------------------------------------------------------------------------
class PretrainRequest(BaseModel):
    dataset_id: int
    lr: float = Field(0.005, gt=0.0, description="Adam learning rate.")
    batch_size: int = Field(64, ge=1)
    epochs: int = Field(8, ge=1, le=200)


class PretrainResponse(BaseModel):
    run_id: int
    val_loss: float = Field(..., description="Held-out next-char cross-entropy.")


# ---------------------------------------------------------------------------
# Fine-tuning  (supervised; pretrained vs scratch)
# ---------------------------------------------------------------------------
class FinetuneRequest(BaseModel):
    dataset_id: int
    pretrain_run_id: Optional[int] = Field(
        None, description="Warm-start from this pretrain run; None trains only scratch."
    )
    n_labeled: int = Field(240, ge=30, le=5000, description="Small labelled train set.")
    lr: float = Field(0.004, gt=0.0)
    batch_size: int = Field(32, ge=1)
    epochs: int = Field(6, ge=1, le=200)
    test_size: float = Field(0.25, gt=0.0, lt=1.0)


class ClassMetrics(BaseModel):
    accuracy: float
    macro_f1: float


class FinetuneResponse(BaseModel):
    run_id: int
    pretrained: Optional[ClassMetrics] = Field(
        None, description="Metrics of the fine-tuned (warm-started) model."
    )
    scratch: ClassMetrics = Field(..., description="Metrics of the from-scratch baseline.")


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
class Run(BaseModel):
    """A training-run row as stored in Supabase (metrics only)."""

    id: int
    dataset_id: int
    run_type: str
    lr: float
    batch_size: int
    epochs: int
    accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    scratch_accuracy: Optional[float] = None
    val_loss: Optional[float] = None
    pretrain_run_id: Optional[int] = None
    created_at: datetime


# ---------------------------------------------------------------------------
# Generation  (temperature sampling from the LM)
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    run_id: int = Field(..., description="A pretrain run whose LM head to sample from.")
    prompt: str = Field("ab", min_length=1, max_length=32, description="Seed characters.")
    length: int = Field(24, ge=1, le=64, description="How many characters to generate.")
    temperature: float = Field(0.8, gt=0.0, le=5.0)
    seed: int = Field(0, description="Sampling seed for reproducibility.")


class GenerateResponse(BaseModel):
    prompt: str
    generated: str = Field(..., description="Only the newly sampled characters.")
    text: str = Field(..., description="Prompt + generated, concatenated.")


# ---------------------------------------------------------------------------
# Prediction  (classify a string's dialect)
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    run_id: int = Field(..., description="A finetune run.")
    sequence: List[int] = Field(..., min_length=2, description="Character ids to classify.")


class PredictResponse(BaseModel):
    run_id: int
    label: int
    class_name: str
    confidence: float
    probs: List[float]
    sequence: List[int]
    sequence_sha256: str = Field(..., description="Hash of the input (no raw chars stored).")
    true_class: Optional[str] = None


class PredictSampleRequest(BaseModel):
    run_id: int
    seed: int = Field(0, description="Seed for the generated sample.")
    true_class: Optional[int] = Field(None, ge=0)


# ---------------------------------------------------------------------------
# Meta / Ops
# ---------------------------------------------------------------------------
class ClassesResponse(BaseModel):
    classes: List[str]
    n_classes: int


class Health(BaseModel):
    status: str
    model_loader: bool
    supabase: bool


class Version(BaseModel):
    git_sha: str
    torch_version: str
    supabase_project_ref: Optional[str]
