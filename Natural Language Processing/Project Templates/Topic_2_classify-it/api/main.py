"""FastAPI service — Cloud #2, deployed on Render.com.

FINISHED CODE, with one exception: the route bodies call into ``api.nlp``, which
is where your work lives. Read this file to see the shape of the service, then go
implement ``api/nlp.py``.

Design rules this file follows, which your report should be able to explain:

* No model logic here. Routes validate, delegate, log, and return. If you find
  yourself calling a tokenizer in this file, it belongs in nlp.py.
* No training here. Not in a route, not at import time, not "just once on
  startup". The web service loads an artifact and answers; training happens
  offline, where there is memory.
* No UI logic here either. The service returns data; the Streamlit tier decides
  what it looks like.
* Logging failures never fail the request (see api/db.py).

Run locally:  uvicorn api.main:app --reload
Then open:    http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import os
import subprocess
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api import db, nlp
from shared.schemas import (
    Health,
    LabelSchema,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictionResult,
    PredictRequest,
    Run,
    RunsResponse,
    Version,
)

SERVICE_NAME = "classify-it-api"

# The build's idea of which model configuration is live. Individual predictions
# carry the version of the ARTIFACT that answered them (nlp.predict fills that
# in), which is the one that matters for auditing; this is the coarse label you
# see on /version when you check that a deploy actually shipped.
MODEL_VERSION = os.environ.get("MODEL_VERSION", "classify-it-v1")

app = FastAPI(
    title="Classify-It API",
    description=(
        "Single-label binary text classification: a TF-IDF + logistic-regression "
        "baseline and a fine-tuned DistilBERT encoder, served side by side."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


def _git_sha() -> str:
    """Short SHA of the running build. Render sets RENDER_GIT_COMMIT for us."""
    if os.environ.get("RENDER_GIT_COMMIT"):
        return os.environ["RENDER_GIT_COMMIT"][:7]
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], timeout=2)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Infrastructure routes — required by the assignment, already working.
# ---------------------------------------------------------------------------
@app.get("/healthz", response_model=Health)
def healthz() -> Health:
    if not db.configured():
        return Health(status="ok", database="not_configured")
    return Health(status="ok", database="ok" if db.ping() else "unreachable")


@app.get("/version", response_model=Version)
def version() -> Version:
    return Version(service=SERVICE_NAME, git_sha=_git_sha(), model_version=MODEL_VERSION)


# ---------------------------------------------------------------------------
# Product routes — the bodies delegate to api/nlp.py (your work).
# ---------------------------------------------------------------------------
@app.get("/schema", response_model=LabelSchema)
def schema() -> LabelSchema:
    """What this classifier predicts: the two labels and the corpus behind them."""
    try:
        return nlp.label_schema()
    except NotImplementedError as exc:
        # A friendlier 501 than a 500 while you are still building.
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictionResult)
def predict(req: PredictRequest) -> PredictionResult:
    """Classify one text and log what was answered."""
    started = time.perf_counter()
    try:
        result = nlp.predict(req.text, req.model_kind)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        # Missing artifact, unknown model kind, label mismatch — all things the
        # caller can act on, so 400 rather than 500.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result.latency_ms = round((time.perf_counter() - started) * 1000, 2)

    row = db.insert_prediction(
        text_sha256=result.text_sha256,
        predicted_label=result.label,
        probability=result.probability,
        model_kind=result.model_kind,
        model_version=result.model_version,
        latency_ms=result.latency_ms,
    )
    if row:
        result.prediction_id = row["id"]
    return result


@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest) -> PredictBatchResponse:
    """Classify a list of texts in one pass and log every one of them.

    Order in equals order out — the response is zipped against the request by
    position on the client side.
    """
    started = time.perf_counter()
    try:
        results = nlp.predict_batch(req.texts, req.model_kind)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if len(results) != len(req.texts):
        # Defensive, and worth keeping: a batch whose length does not match the
        # request will mislabel the caller's spreadsheet without any error.
        raise HTTPException(
            status_code=500,
            detail=(
                f"predict_batch returned {len(results)} results for "
                f"{len(req.texts)} inputs. Results must be one-per-input, in order."
            ),
        )

    elapsed_ms = (time.perf_counter() - started) * 1000
    per_item = round(elapsed_ms / len(results), 2) if results else None
    for r in results:
        r.latency_ms = per_item

    rows = db.insert_predictions(
        [
            {
                "text_sha256": r.text_sha256,
                "predicted_label": r.label,
                "probability": r.probability,
                "model_kind": r.model_kind,
                "model_version": r.model_version,
                "latency_ms": r.latency_ms,
            }
            for r in results
        ]
    )
    for r, row in zip(results, rows):
        r.prediction_id = row.get("id")

    return PredictBatchResponse(predictions=results, count=len(results))


@app.get("/runs", response_model=RunsResponse)
def runs(limit: int = 50) -> RunsResponse:
    """Recent TRAINING runs, newest first — the data source for the
    Baseline vs. Transformer tab.

    Served predictions are not here; they are in the predictions table, which the
    UI reads directly with the anon key.
    """
    return RunsResponse(runs=[Run(**r) for r in db.latest_runs(limit=limit)])
