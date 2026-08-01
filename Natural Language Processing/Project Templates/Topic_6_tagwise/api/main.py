"""FastAPI service — Cloud #2, deployed on Render.com.

FINISHED CODE, with one exception: the /tag route calls into ``api.nlp``, which
is where your work lives. Read this file to see the shape of the service, then go
implement ``api/nlp.py``.

Design rules this file follows, which your report should be able to explain:

* No NLP logic here. Routes validate, delegate, log, and return. If you find
  yourself writing a suffix rule in this file, it belongs in nlp.py.
* No UI logic here either. The service returns tokens and tags; the Streamlit
  tier decides what colour a VERB is.
* No training here. Fine-tuning happens offline in your own script, which writes
  a row to ``runs``. A web request loads a saved model; it never builds one.
  A route that trains would time out on Render's free plan and would also mean
  two concurrent requests training two different models at once.
* Logging failures never fail the request (see api/db.py).

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
    Health,
    Run,
    RunsResponse,
    TagRequest,
    TagResponse,
    Version,
)

SERVICE_NAME = "tagwise-api"

# Bump this whenever you rebuild a tagger — a new lookup table, a new fine-tune,
# a changed fallback rule. Every logged run and every logged tagging carries it,
# which is what lets you say "accuracy moved because of v3, not because of the
# input" when you compare two rows later.
MODEL_VERSION = os.environ.get("MODEL_VERSION", "tagwise-v1")

app = FastAPI(
    title="TagWise API",
    description="Part-of-speech tagging for the three-cloud stack: a most-frequent-tag "
    "lookup baseline and a fine-tuned transformer token classifier.",
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
@app.post("/tag", response_model=TagResponse)
def tag(req: TagRequest) -> TagResponse:
    """Tag one sentence with either model, and log the request.

    Note what gets logged: the hash of the sentence, the tag sequence, the model,
    the version, the timestamp. Not the sentence. You can still answer "how many
    requests did the transformer serve last week" and "did this exact sentence
    come through twice" without holding anyone's text.
    """
    try:
        result = nlp.tag_sentence(req.sentence, req.model)
    except NotImplementedError as exc:
        # A friendlier 501 than a 500 while you are still building.
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        # e.g. "no fine-tuned model found at models/tagwise-transformer"
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result.model_version = MODEL_VERSION
    row = db.insert_tagging(
        sentence_sha256=nlp.sha256_text(req.sentence),
        token_count=len(result.tokens),
        tag_sequence=result.tag_sequence,
        model=req.model,
        model_version=MODEL_VERSION,
        unknown_count=result.unknown_count,
    )
    if row:
        result.tagging_id = row["id"]
    return result


@app.get("/runs", response_model=RunsResponse)
def runs(limit: int = 50) -> RunsResponse:
    """Recent tagger builds, newest first.

    This is what the 'Baseline vs. Transformer' tab reads: accuracy, macro-F1,
    and the stored confusion matrix for each model. There is no /evaluate
    endpoint on purpose — evaluation happens once, offline, against a fixed
    held-out split, and the result is a fact about a build. Recomputing it per
    page load would be slow and would invite the temptation to evaluate on
    whatever data happened to be handy.
    """
    return RunsResponse(runs=[Run(**r) for r in db.latest_runs(limit=limit)])
