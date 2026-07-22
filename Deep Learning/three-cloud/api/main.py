"""FastAPI model service -- Cloud #2 (deployed on Render.com).

Responsibilities:
  * expose the model behind HTTP so the Streamlit UI can call it over HTTPS,
  * read datasets from Supabase before training,
  * write run rows and prediction rows back to Supabase.

There is NO UI code here and NO business logic in the UI -- separation of
concerns across the three clouds.
"""
from __future__ import annotations

import os
import subprocess

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api import db
from api.training import predict as run_predict
from api.training import train_linear_regression
from shared.data import generate_linear
from shared.schemas import (
    Dataset,
    DatasetCreate,
    Health,
    Metrics,
    PredictRequest,
    PredictResponse,
    Run,
    TrainRequest,
    TrainResponse,
    Version,
)

app = FastAPI(
    title="Regress-It API",
    description="Linear-regression model service for the three-cloud stack.",
    version="1.0.0",
)

# The UI lives on a different origin (Streamlit Cloud), so CORS must allow it.
# "*" is fine for a teaching demo; tighten to your Streamlit URL in production.
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
    # https://<ref>.supabase.co -> <ref>
    if url.startswith("https://"):
        return url.split("//", 1)[1].split(".", 1)[0]
    return None


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
@app.post("/datasets", response_model=Dataset, tags=["datasets"])
def create_dataset(req: DatasetCreate) -> Dataset:
    """Generate a synthetic dataset and persist it to Supabase."""
    xs, ys = generate_linear(req.slope, req.intercept, req.noise, req.n_points)
    row = db.insert_dataset(
        req.name, req.slope, req.intercept, req.noise, req.n_points, xs, ys
    )
    return Dataset(**{k: row[k] for k in Dataset.model_fields})


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
@app.post("/train", response_model=TrainResponse, tags=["training"])
def train(req: TrainRequest) -> TrainResponse:
    """Read a dataset from Supabase, train, write the run row, return metrics."""
    dataset = db.get_dataset(req.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset_id not found")

    metrics, weights, _loss = train_linear_regression(
        dataset["xs"],
        dataset["ys"],
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
        mse=metrics["mse"],
        mae=metrics["mae"],
        r2=metrics["r2"],
        weights_json=weights,
    )
    return TrainResponse(run_id=run["id"], metrics=Metrics(**metrics), weights=weights)


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
@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(req: PredictRequest) -> PredictResponse:
    """Predict yhat for a feature value using a stored run; log to Supabase."""
    run = db.get_run(req.run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    yhat = run_predict(run["weights_json"], req.x)
    db.insert_prediction(req.run_id, req.x, yhat)
    return PredictResponse(run_id=req.run_id, x=req.x, yhat=yhat)


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
        supabase_project_ref=_project_ref(),
    )
