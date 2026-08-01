"""FastAPI service — Cloud #2, deployed on Render.com.

FINISHED CODE, with one exception: the route bodies call into ``api.nlp``, which
is where your work lives. Read this file to see the shape of the service, then go
implement ``api/nlp.py``.

Design rules this file follows, which your report should be able to explain:

* No NLP logic here. Routes validate, delegate, log, and return. If you find
  yourself writing a sampling loop in this file, it belongs in nlp.py.
* No UI logic here either. The service returns data; the Streamlit tier decides
  what it looks like.
* No training here. Fine-tuning happens offline (see api/nlp.fine_tune); this
  process only loads a checkpoint and generates from it.
* Logging failures never fail the request (see api/db.py).

Endpoints:
    POST /generate   generate text under an explicit decoding strategy
    POST /rate       write a human quality rating back to a generation
    GET  /history    recent generations, newest first
    GET  /healthz    liveness plus database and model status
    GET  /version    build identity

Run locally:  uvicorn api.main:app --reload
Then open:    http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import os
import subprocess

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api import db, nlp
from shared.schemas import (
    Generation,
    GenerateRequest,
    GenerateResponse,
    Health,
    HistoryResponse,
    RateRequest,
    RateResponse,
    Version,
)

SERVICE_NAME = "gentext-api"

# The checkpoint this instance serves. Point it at your fine-tuned model on the
# Hugging Face Hub once you have published one; the base model is the default so
# the service works before you have trained anything.
BASE_MODEL = os.environ.get("MODEL_NAME", nlp.DEFAULT_MODEL)

# Bump this whenever the weights or the decoding defaults change. Every logged
# generation carries it, and it is the join key back to the training_runs row —
# which is how you say "quality improved in v3" and can prove which v3 that was.
MODEL_VERSION = os.environ.get("MODEL_VERSION", "gentext-v1")

app = FastAPI(
    title="GenText API",
    description="Controllable text generation for the three-cloud stack.",
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
    """Liveness. Deliberately does NOT load the model.

    A health check that loads half a gigabyte of weights will time out on the
    free plan and Render will restart the service in a loop. The ``model_loaded``
    flag reports whether a decoder is already resident so you can tell a cold
    start from a crash.
    """
    database = "not_configured"
    if db.configured():
        database = "ok" if db.ping() else "unreachable"
    return Health(status="ok", database=database, model_loaded=nlp.model_is_loaded())


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
@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """Generate a continuation under an explicit decoding strategy, and log it."""
    try:
        result = nlp.generate(req.prompt, req.params)
    except NotImplementedError as exc:
        # A friendlier 501 than a 500 while you are still building.
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        # Unknown model id, incompatible parameter combination, and so on.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result.model_version = MODEL_VERSION

    row = db.insert_generation(
        prompt_sha256=result.prompt_sha256,
        strategy=result.strategy,
        decoding_params=result.params.model_dump(),
        generated_text=result.generated_text,
        model_version=MODEL_VERSION,
        prompt_token_count=result.prompt_token_count,
        generated_token_count=result.generated_token_count,
        distinct_1=result.distinct_1,
        distinct_2=result.distinct_2,
        perplexity=result.perplexity,
        latency_ms=result.latency_ms,
    )
    if row:
        result.generation_id = row["id"]
    return result


@app.post("/rate", response_model=RateResponse)
def rate(req: RateRequest) -> RateResponse:
    """Attach one rater's independent score to an existing generation.

    This is the write path the human evaluation depends on. Two raters submitting
    for the same generation_id produce two entries in the row's ``ratings``
    array — the disagreement between them is a required part of the report, so
    neither one overwrites the other.
    """
    try:
        payload = nlp.record_rating(
            generation_id=req.generation_id,
            rater_id=req.rater_id,
            rating=req.rating,
            dimensions={
                k: v
                for k, v in {
                    "coherence": req.coherence,
                    "fluency": req.fluency,
                    "relevance": req.relevance,
                }.items()
                if v is not None
            },
            notes=req.notes,
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not db.configured():
        return RateResponse(
            generation_id=req.generation_id,
            human_rating=req.rating,
            rating_count=1,
            stored=False,
        )

    row = db.append_rating(req.generation_id, payload)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No generation with id {req.generation_id}, or the write failed.",
        )
    return RateResponse(
        generation_id=req.generation_id,
        human_rating=row.get("human_rating"),
        rating_count=len(row.get("ratings") or []),
        stored=True,
    )


@app.get("/history", response_model=HistoryResponse)
def history(limit: int = 50, strategy: str | None = None) -> HistoryResponse:
    """Recent generations, newest first — the data source for the History tab.

    Note that this reads the generations table, not training_runs. A project that
    logged only its fine-tuning runs would have three rows here and an empty tab.
    """
    rows = db.latest_generations(limit=limit)
    if strategy:
        rows = [r for r in rows if r.get("strategy") == strategy]
    return HistoryResponse(generations=[Generation(**r) for r in rows])
