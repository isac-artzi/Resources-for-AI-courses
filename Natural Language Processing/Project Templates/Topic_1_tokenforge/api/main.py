"""FastAPI service — Cloud #2, deployed on Render.com.

FINISHED CODE, with one exception: the two route bodies call into ``api.nlp``,
which is where your work lives. Read this file to see the shape of the service,
then go implement ``api/nlp.py``.

Design rules this file follows, which your report should be able to explain:

* No NLP logic here. Routes validate, delegate, log, and return. If you find
  yourself writing a regex in this file, it belongs in nlp.py.
* No UI logic here either. The service returns data; the Streamlit tier decides
  what it looks like.
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
    PreprocessRequest,
    PreprocessResponse,
    Run,
    RunsResponse,
    TokenizeRequest,
    TokenizeResponse,
    Version,
)

SERVICE_NAME = "tokenforge-api"

# Bump this whenever you change the preprocessing rules or the tokenizer set.
# Every logged run carries it, which is what lets you say "the OOV rate dropped
# because of v3, not because of the input" when you compare runs later.
MODEL_VERSION = os.environ.get("MODEL_VERSION", "tokenforge-v1")

app = FastAPI(
    title="TokenForge API",
    description="Text preprocessing and subword tokenization for the three-cloud stack.",
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
@app.post("/preprocess", response_model=PreprocessResponse)
def preprocess(req: PreprocessRequest) -> PreprocessResponse:
    """Classical preprocessing with a full audit trail of what ran."""
    try:
        result = nlp.preprocess(req.text, req.options)
    except NotImplementedError as exc:
        # A friendlier 501 than a 500 while you are still building.
        raise HTTPException(status_code=501, detail=str(exc)) from exc

    row = db.insert_run(
        kind="preprocess",
        text_sha256=nlp.sha256_text(req.text),
        config=req.options.model_dump(),
        model_version=MODEL_VERSION,
        token_count_before=result.token_count_before,
        token_count_after=result.token_count_after,
    )
    if row:
        result.run_id = row["id"]
    return result


@app.post("/tokenize", response_model=TokenizeResponse)
def tokenize(req: TokenizeRequest) -> TokenizeResponse:
    """Run the same text through several subword tokenizers and compare them."""
    try:
        results = [nlp.subword_tokenize(req.text, name) for name in req.tokenizers]
        overlap = nlp.vocabulary_overlap(results)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    worst_oov = max((r.oov_rate for r in results), default=None)
    row = db.insert_run(
        kind="tokenize",
        text_sha256=nlp.sha256_text(req.text),
        config={"tokenizers": req.tokenizers},
        model_version=MODEL_VERSION,
        token_count_before=len(req.text.split()),
        token_count_after=results[0].token_count if results else None,
        oov_rate=worst_oov,
    )
    return TokenizeResponse(
        text=req.text,
        results=results,
        vocabulary_overlap=overlap,
        run_id=row["id"] if row else None,
    )


@app.get("/runs", response_model=RunsResponse)
def runs(limit: int = 50) -> RunsResponse:
    """Recent runs, newest first — the data source for the History tab."""
    return RunsResponse(runs=[Run(**r) for r in db.latest_runs(limit=limit)])
