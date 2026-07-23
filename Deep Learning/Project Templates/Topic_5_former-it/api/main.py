"""FastAPI model service -- Cloud #2 (deployed on Render.com).

Responsibilities:
  * expose the Transformer classifier behind HTTP so the Streamlit UI can call it
    over HTTPS,
  * read dataset parameters from Supabase before training,
  * write run rows, model artifacts, and sequence metadata (hash only) back.

There is NO UI code here and NO business logic in the UI -- separation of
concerns across the three clouds. This is "Former-It": sequence classification
with a from-scratch tiny Transformer encoder that returns per-head self-attention
matrices so a client can see *which positions attend to which*.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api import db
from api.training import predict_seq, train_transformer
from shared.data import CLASS_NAMES, N_CLASSES, VOCAB_SIZE, generate_one
from shared.schemas import (
    ClassesResponse,
    ClassMetrics,
    Dataset,
    DatasetCreate,
    Health,
    PredictRequest,
    PredictResponse,
    PredictSampleRequest,
    Run,
    TrainRequest,
    TrainResponse,
    Version,
)

app = FastAPI(
    title="Former-It API",
    description="From-scratch Transformer encoder classifier for the three-cloud stack.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


def _git_sha() -> str:
    if os.environ.get("RENDER_GIT_COMMIT"):
        return os.environ["RENDER_GIT_COMMIT"][:7]
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _project_ref() -> str | None:
    url = os.environ.get("SUPABASE_URL", "")
    if url.startswith("https://"):
        return url.split("//", 1)[1].split(".", 1)[0]
    return None


def _artifact_or_404(run_id: int) -> str:
    model_b64 = db.get_run_artifact(run_id)
    if model_b64 is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    return model_b64


def _seq_sha256(sequence: list[int]) -> str:
    """Stable hash of the symbol sequence (no raw symbols are ever stored)."""
    return hashlib.sha256(json.dumps(sequence).encode()).hexdigest()


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
@app.post("/datasets", response_model=Dataset, tags=["datasets"])
def create_dataset(req: DatasetCreate) -> Dataset:
    """Register a synthetic sequence dataset (parameters only; symbols come from seed)."""
    row = db.insert_dataset(
        name=req.name,
        n_rows=req.n_rows,
        seq_len=req.seq_len,
        vocab_size=VOCAB_SIZE,
        n_classes=N_CLASSES,
        seed=req.seed,
    )
    return Dataset(**{k: row[k] for k in Dataset.model_fields})


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
@app.post("/train", response_model=TrainResponse, tags=["training"])
def train(req: TrainRequest) -> TrainResponse:
    """Read dataset params from Supabase, train the model, persist run + artifact."""
    dataset = db.get_dataset(req.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset_id not found")

    metrics, model_b64, _loss = train_transformer(
        n_rows=dataset["n_rows"],
        seq_len=dataset["seq_len"],
        seed=dataset["seed"],
        lr=req.lr,
        batch_size=req.batch_size,
        epochs=req.epochs,
        test_size=req.test_size,
    )
    run = db.insert_run(
        dataset_id=req.dataset_id,
        lr=req.lr,
        batch_size=req.batch_size,
        epochs=req.epochs,
        metrics=metrics,
        model_b64=model_b64,
    )
    return TrainResponse(run_id=run["id"], metrics=ClassMetrics(**metrics))


@app.get("/runs/{run_id}", response_model=Run, tags=["training"])
def get_run(run_id: int) -> Run:
    row = db.get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    return Run(**{k: row[k] for k in Run.model_fields})


@app.get("/runs", response_model=list[Run], tags=["training"])
def list_runs() -> list[Run]:
    rows = db.latest_runs(limit=50)
    return [Run(**{k: r[k] for k in Run.model_fields}) for r in rows]


# ---------------------------------------------------------------------------
# prediction  (per-head self-attention explainability)
# ---------------------------------------------------------------------------
def _predict_sequence(
    run_id: int, sequence: list[int], true_class: str | None
) -> PredictResponse:
    model_b64 = _artifact_or_404(run_id)
    result = predict_seq(model_b64, sequence)
    sha256 = _seq_sha256(sequence)
    db.insert_sequence_metadata(
        run_id=run_id,
        sha256=sha256,
        length=len(sequence),
        label=result["label"],
        class_name=result["class_name"],
        confidence=result["confidence"],
    )
    return PredictResponse(
        run_id=run_id,
        sequence=sequence,
        sequence_sha256=sha256,
        true_class=true_class,
        **result,
    )


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(req: PredictRequest) -> PredictResponse:
    """Classify a symbol sequence and return per-layer, per-head attention.

    Only the sequence's sha256 hash + length + prediction are logged -- never
    the raw symbols.
    """
    if any(t < 0 or t >= VOCAB_SIZE for t in req.sequence):
        raise HTTPException(
            status_code=422, detail=f"symbol ids must be in [0, {VOCAB_SIZE})"
        )
    return _predict_sequence(req.run_id, req.sequence, true_class=None)


@app.post("/predict_sample", response_model=PredictResponse, tags=["prediction"])
def predict_sample(req: PredictSampleRequest) -> PredictResponse:
    """Generate a synthetic sequence, classify it, and return attention matrices.

    Lets the UI demonstrate the model without the user typing a sequence.
    """
    kind = (
        req.true_class
        if (req.true_class is None or req.true_class < N_CLASSES)
        else None
    )
    seq, label = generate_one(seed=req.seed, kind=kind)
    return _predict_sequence(
        req.run_id, [int(t) for t in seq], true_class=CLASS_NAMES[label]
    )


# ---------------------------------------------------------------------------
# meta
# ---------------------------------------------------------------------------
@app.get("/classes", response_model=ClassesResponse, tags=["meta"])
def classes() -> ClassesResponse:
    """Return the class labels the model predicts over."""
    return ClassesResponse(classes=CLASS_NAMES, n_classes=N_CLASSES)


# ---------------------------------------------------------------------------
# ops
# ---------------------------------------------------------------------------
@app.get("/healthz", response_model=Health, tags=["ops"])
def healthz() -> Health:
    supabase_ok = db.ping()
    model_ok = torch.tensor([1.0]).sum().item() == 1.0
    status = "ok" if (supabase_ok and model_ok) else "degraded"
    return Health(status=status, model_loader=model_ok, supabase=supabase_ok)


@app.get("/version", response_model=Version, tags=["ops"])
def version() -> Version:
    return Version(
        git_sha=_git_sha(),
        torch_version=torch.__version__,
        supabase_project_ref=_project_ref(),
    )
