"""FastAPI service — Cloud #2, deployed on Render.com.

FINISHED CODE, with one exception: the route bodies call into ``api.nlp``, which
is where your work lives. Read this file to see the shape of the service, then
go implement ``api/nlp.py``.

Design rules this file follows, which your report should be able to explain:

* No NLP logic here. Routes validate, delegate, log, and return. If you find
  yourself writing a threshold comparison in this file, it belongs in nlp.py.
* No UI logic here either. The service returns numbers; the Streamlit tier
  decides whether they are a bar chart or a table.
* Logging failures never fail the request (see api/db.py).
* Every prediction that goes out gets written down, with the model version that
  produced it. That is not bookkeeping, it is the audit requirement in the
  product brief, and it is why /predict does not have a "don't log this" flag.

Run locally:  uvicorn api.main:app --reload
Then open:    http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import os
import subprocess
from collections import Counter
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api import db, nlp
from shared.schemas import (
    AuditResponse,
    Health,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictionRecord,
    PredictRequest,
    PredictResponse,
    Version,
)

SERVICE_NAME = "moodlens-api"

# Bump this whenever you retrain or change the serving configuration. Every
# logged prediction carries it, which is what lets you say "the false negatives
# on short reviews came from v2, and v3 fixed them" instead of guessing.
MODEL_VERSION = os.environ.get("MODEL_VERSION", "moodlens-v1")

# The checkpoint you fine-tuned FROM. Reported by /version because a model card
# that does not name its base model is not a model card.
BASE_MODEL = os.environ.get("BASE_MODEL", "distilbert-base-uncased")

app = FastAPI(
    title="MoodLens API",
    description="Document and aspect-based sentiment for the three-cloud stack.",
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


def _model_loaded() -> bool:
    """True once the classifier is in memory, without forcing it to load.

    Deliberately does NOT call load_classifier(). /healthz is what Render polls;
    a health check that pulls half a gigabyte of weights into RAM to answer is a
    health check that fails the deploy it was meant to protect.
    """
    try:
        return bool(getattr(nlp, "_CLASSIFIER", None))
    except Exception:
        return False


def _score(text: str, include_aspects: bool) -> PredictResponse:
    """Score one text and log it. Shared by /predict and /predict_batch."""
    sentiment = nlp.predict_sentiment(text)
    aspects = nlp.extract_aspects(text) if include_aspects else []
    digest = nlp.sha256_text(text)

    row = db.insert_prediction(
        text_sha256=digest,
        char_count=len(text),
        label=sentiment.label,
        probability_positive=sentiment.probability_positive,
        confidence=sentiment.confidence,
        calibrated=sentiment.calibrated,
        aspects=[a.model_dump() for a in aspects],
        model_name=sentiment.model_name,
        model_version=MODEL_VERSION,
    )
    return PredictResponse(
        text_sha256=digest,
        char_count=len(text),
        sentiment=sentiment,
        aspects=aspects,
        model_version=MODEL_VERSION,
        prediction_id=row["id"] if row else None,
    )


# ---------------------------------------------------------------------------
# Infrastructure routes — required by the assignment, already working.
# ---------------------------------------------------------------------------
@app.get("/healthz", response_model=Health)
def healthz() -> Health:
    if not db.configured():
        return Health(status="ok", database="not_configured", model_loaded=_model_loaded())
    return Health(
        status="ok",
        database="ok" if db.ping() else "unreachable",
        model_loaded=_model_loaded(),
    )


@app.get("/version", response_model=Version)
def version() -> Version:
    return Version(
        service=SERVICE_NAME,
        git_sha=_git_sha(),
        model_version=MODEL_VERSION,
        base_model=BASE_MODEL,
    )


# ---------------------------------------------------------------------------
# Product routes — the bodies delegate to api/nlp.py (your work).
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Score one review: document sentiment plus the aspect breakdown."""
    try:
        return _score(req.text, req.include_aspects)
    except NotImplementedError as exc:
        # A friendlier 501 than a 500 while you are still building.
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest) -> PredictBatchResponse:
    """Score up to 64 reviews in one call.

    Note the shape: the batch is scored with ONE call into
    ``predict_sentiment_batch`` so your implementation can run a single forward
    pass, and only the aspect step (which is optional here) is per item.
    """
    try:
        sentiments = nlp.predict_sentiment_batch(req.texts)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if len(sentiments) != len(req.texts):
        raise HTTPException(
            status_code=500,
            detail=(
                f"predict_sentiment_batch returned {len(sentiments)} results for "
                f"{len(req.texts)} inputs — results must come back in input order, "
                "one per input."
            ),
        )

    results: List[PredictResponse] = []
    pending: List[dict] = []
    for text, sentiment in zip(req.texts, sentiments):
        try:
            aspects = nlp.extract_aspects(text) if req.include_aspects else []
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        digest = nlp.sha256_text(text)
        pending.append(
            {
                "text_sha256": digest,
                "char_count": len(text),
                "label": sentiment.label,
                "probability_positive": sentiment.probability_positive,
                "confidence": sentiment.confidence,
                "calibrated": sentiment.calibrated,
                "aspects": [a.model_dump() for a in aspects],
                "model_name": sentiment.model_name,
                "model_version": MODEL_VERSION,
            }
        )
        results.append(
            PredictResponse(
                text_sha256=digest,
                char_count=len(text),
                sentiment=sentiment,
                aspects=aspects,
                model_version=MODEL_VERSION,
            )
        )

    # One round trip for the whole batch (see api/db.py).
    rows = db.insert_predictions(pending)
    for result, row in zip(results, rows):
        result.prediction_id = row.get("id")

    return PredictBatchResponse(
        results=results,
        count=len(results),
        label_counts=dict(Counter(r.sentiment.label for r in results)),
    )


@app.get("/audit", response_model=AuditResponse)
def audit(
    limit: int = Query(default=100, ge=1, le=500),
    label: Optional[str] = Query(default=None, description='Filter, e.g. "negative"'),
) -> AuditResponse:
    """Recent served predictions, newest first — the audit trail.

    Read-only and hash-only. There is no endpoint that returns the text of a
    logged review, because the text was never stored.
    """
    rows = db.latest_predictions(limit=limit, label=label)
    records = [PredictionRecord(**r) for r in rows]
    return AuditResponse(predictions=records, count=len(records))
