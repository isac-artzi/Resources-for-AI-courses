"""FastAPI model service -- Cloud #2 (deployed on Render.com).

Responsibilities:
  * expose the MLP behind HTTP so the Streamlit UI can call it over HTTPS,
  * read datasets from Supabase before training,
  * write run rows, model artifacts, and prediction rows back to Supabase.

There is NO UI code here and NO business logic in the UI -- separation of
concerns across the three clouds. This is "Income-Insight": tabular binary
classification with a PyTorch MLP + sklearn preprocessing pipeline.
"""
from __future__ import annotations

import os
import subprocess
from collections import defaultdict

import sklearn
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api import db
from api.training import predict_records, train_income_classifier
from shared.data import (
    CATEGORICAL_COLS,
    CATEGORIES,
    FEATURE_COLS,
    NUMERIC_COLS,
    TARGET_CLASSES,
    TARGET_NAME,
    generate_tabular,
)
from shared.schemas import (
    AuditGroup,
    AuditResponse,
    BatchItem,
    ClassMetrics,
    Dataset,
    DatasetCreate,
    Health,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
    Run,
    SchemaResponse,
    TrainRequest,
    TrainResponse,
    Version,
)

app = FastAPI(
    title="Income-Insight API",
    description="Tabular MLP classifier for the three-cloud stack.",
    version="1.0.0",
)

# The UI lives on a different origin (Streamlit Cloud), so CORS must allow it.
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


def _income(label: int) -> str:
    return TARGET_CLASSES[label]


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
@app.post("/datasets", response_model=Dataset, tags=["datasets"])
def create_dataset(req: DatasetCreate) -> Dataset:
    """Generate a synthetic tabular dataset and persist it to Supabase."""
    records, labels = generate_tabular(req.n_rows, noise=req.noise, seed=req.seed)
    positive_rate = sum(labels) / len(labels)
    row = db.insert_dataset(
        name=req.name,
        n_rows=req.n_rows,
        n_features=len(FEATURE_COLS),
        positive_rate=positive_rate,
        records=records,
        labels=labels,
    )
    return Dataset(**{k: row[k] for k in Dataset.model_fields})


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
@app.post("/train", response_model=TrainResponse, tags=["training"])
def train(req: TrainRequest) -> TrainResponse:
    """Read a dataset from Supabase, train the MLP, persist run + artifact."""
    dataset = db.get_dataset(req.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset_id not found")

    metrics, model_b64, _loss = train_income_classifier(
        dataset["records"],
        dataset["labels"],
        hidden_dim=req.hidden_dim,
        lr=req.lr,
        batch_size=req.batch_size,
        epochs=req.epochs,
        test_size=req.test_size,
    )
    run = db.insert_run(
        dataset_id=req.dataset_id,
        hidden_dim=req.hidden_dim,
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
    """Return the latest 50 runs from Supabase."""
    rows = db.latest_runs(limit=50)
    return [Run(**{k: r[k] for k in Run.model_fields}) for r in rows]


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------
def _load_artifact_or_404(run_id: int) -> str:
    model_b64 = db.get_run_artifact(run_id)
    if model_b64 is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    return model_b64


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(req: PredictRequest) -> PredictResponse:
    """Classify one record using a stored run; log it to Supabase."""
    model_b64 = _load_artifact_or_404(req.run_id)
    label, proba = predict_records(model_b64, [req.features])[0]
    db.insert_prediction(req.run_id, req.features, proba, label)
    return PredictResponse(
        run_id=req.run_id, label=label, income=_income(label), proba=proba
    )


@app.post("/predict_batch", response_model=PredictBatchResponse, tags=["prediction"])
def predict_batch(req: PredictBatchRequest) -> PredictBatchResponse:
    """Classify many records at once; each is logged to Supabase."""
    if not req.records:
        raise HTTPException(status_code=422, detail="records must be non-empty")
    model_b64 = _load_artifact_or_404(req.run_id)
    results = predict_records(model_b64, req.records)
    items = []
    for features, (label, proba) in zip(req.records, results):
        db.insert_prediction(req.run_id, features, proba, label)
        items.append(BatchItem(label=label, income=_income(label), proba=proba))
    return PredictBatchResponse(run_id=req.run_id, predictions=items)


# ---------------------------------------------------------------------------
# schema & audit
# ---------------------------------------------------------------------------
@app.get("/schema", response_model=SchemaResponse, tags=["meta"])
def schema() -> SchemaResponse:
    """Expose the feature contract so the UI can build its form dynamically."""
    return SchemaResponse(
        numeric_features=NUMERIC_COLS,
        categorical_features=CATEGORICAL_COLS,
        categories=CATEGORIES,
        target_name=TARGET_NAME,
        target_classes=TARGET_CLASSES,
    )


@app.get("/audit", response_model=AuditResponse, tags=["meta"])
def audit(run_id: int, by: str) -> AuditResponse:
    """Group a run's logged predictions by a categorical feature and report the
    positive-prediction rate per group -- a simple fairness/monitoring view."""
    if by not in CATEGORICAL_COLS:
        raise HTTPException(
            status_code=422, detail=f"'by' must be one of {CATEGORICAL_COLS}"
        )
    rows = db.predictions_for_run(run_id)
    counts: dict[str, int] = defaultdict(int)
    positives: dict[str, int] = defaultdict(int)
    for r in rows:
        group = str(r["features"].get(by, "unknown"))
        counts[group] += 1
        positives[group] += int(r["label"])
    groups = [
        AuditGroup(group=g, n=counts[g], positive_rate=positives[g] / counts[g])
        for g in sorted(counts)
    ]
    total = sum(counts.values())
    overall = sum(positives.values()) / total if total else 0.0
    return AuditResponse(
        run_id=run_id, by=by, total=total, overall_positive_rate=overall, groups=groups
    )


# ---------------------------------------------------------------------------
# ops
# ---------------------------------------------------------------------------
@app.get("/healthz", response_model=Health, tags=["ops"])
def healthz() -> Health:
    """200 when the model loader (torch) and the Supabase client are reachable."""
    supabase_ok = db.ping()
    model_ok = torch.tensor([1.0]).sum().item() == 1.0
    status = "ok" if (supabase_ok and model_ok) else "degraded"
    return Health(status=status, model_loader=model_ok, supabase=supabase_ok)


@app.get("/version", response_model=Version, tags=["ops"])
def version() -> Version:
    return Version(
        git_sha=_git_sha(),
        torch_version=torch.__version__,
        sklearn_version=sklearn.__version__,
        supabase_project_ref=_project_ref(),
    )
