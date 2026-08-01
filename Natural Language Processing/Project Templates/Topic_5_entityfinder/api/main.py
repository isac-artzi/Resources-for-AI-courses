"""FastAPI service — Cloud #2, deployed on Render.com.

FINISHED CODE, with one exception: /extract calls into ``api.nlp``, which is
where your work lives. Read this file to see the shape of the service, then go
implement ``api/nlp.py``.

Design rules this file follows, which your report should be able to explain:

* No NLP logic here. Routes validate, delegate, log, and return. If you find
  yourself writing a tag-decoding loop in this file, it belongs in nlp.py.
* No UI logic here either. The service returns data; the Streamlit tier decides
  what it looks like.
* Logging failures never fail the request (see api/db.py).
* /review reads the prediction before it writes the decision, so the reviewer's
  row carries what the model said. The API is where that guarantee lives,
  because the browser cannot be trusted to send it honestly.

Run locally:  uvicorn api.main:app --reload
Then open:    http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import os
import subprocess
import time

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api import db, nlp
from shared.schemas import (
    ExtractRequest,
    ExtractResponse,
    Health,
    ReviewQueueItem,
    ReviewQueueResponse,
    ReviewRequest,
    ReviewResponse,
    Run,
    RunsResponse,
    Version,
)

SERVICE_NAME = "entityfinder-api"

# Bump this whenever you retrain, change the label set, or change the decoding
# rules. Every extraction row carries it, which is what lets you say "recall
# improved because of v3, not because the documents changed" three weeks later.
MODEL_VERSION = os.environ.get("MODEL_VERSION", "entityfinder-v1")

# The cutoff that fills the review queue. Your team's number: set it high and
# reviewers drown, set it low and the errors worth catching never surface.
# It is an environment variable so you can retune it without a redeploy.
CONFIDENCE_THRESHOLD = float(
    os.environ.get("CONFIDENCE_THRESHOLD", str(nlp.DEFAULT_CONFIDENCE_THRESHOLD))
)

app = FastAPI(
    title="EntityFinder API",
    description="Named-entity recognition with a transformer and a CRF baseline.",
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
# Product routes
# ---------------------------------------------------------------------------
@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest) -> ExtractResponse:
    """Extract entities, log the extraction, and hand back ids the queue can use."""
    started = time.perf_counter()
    try:
        entities = nlp.extract_entities(req.text, req.model)
    except NotImplementedError as exc:
        # A friendlier 501 than a 500 while you are still building.
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    latency_ms = int((time.perf_counter() - started) * 1000)

    text_sha256 = nlp.sha256_text(req.text)
    extraction = db.insert_extraction(
        text_sha256=text_sha256,
        model=req.model,
        model_version=MODEL_VERSION,
        entity_count=len(entities),
        latency_ms=latency_ms,
    )

    if extraction:
        rows = db.insert_entities(
            extraction["id"], [e.model_dump(exclude={"entity_id"}) for e in entities]
        )
        # Attach the row ids so the UI can send a review straight back for any
        # of these spans. Supabase returns inserted rows in insertion order.
        for entity, row in zip(entities, rows):
            entity.entity_id = row.get("id")

    return ExtractResponse(
        text=req.text,
        text_sha256=text_sha256,
        model=req.model,
        model_version=MODEL_VERSION,
        entities=entities,
        entity_count=len(entities),
        latency_ms=latency_ms,
        extraction_id=extraction["id"] if extraction else None,
    )


@app.get("/review_queue", response_model=ReviewQueueResponse)
def review_queue(
    threshold: float = Query(
        default=CONFIDENCE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Entities scoring below this go in the queue",
    ),
    limit: int = Query(default=50, ge=1, le=500),
    include_reviewed: bool = False,
) -> ReviewQueueResponse:
    """Low-confidence predictions queued for a human, lowest confidence first."""
    rows = db.review_queue(threshold=threshold, limit=limit, include_reviewed=include_reviewed)
    items = []
    for row in rows:
        parent = row.get("extractions") or {}
        items.append(
            ReviewQueueItem(
                entity_id=row["id"],
                extraction_id=row["extraction_id"],
                text=row["text"],
                start_char=row["start_char"],
                end_char=row["end_char"],
                entity_type=row["entity_type"],
                confidence=row["confidence"],
                model=parent.get("model"),
                model_version=parent.get("model_version"),
                context=row.get("context"),
                created_at=row.get("created_at"),
            )
        )
    return ReviewQueueResponse(threshold=threshold, count=len(items), items=items)


@app.post("/review", response_model=ReviewResponse)
def review(req: ReviewRequest) -> ReviewResponse:
    """Write a reviewer's decision back to Supabase, preserving the prediction.

    The original type, span, and confidence are read from the entities row here
    on the server rather than accepted from the request body. A client that
    supplied its own "original" could rewrite history by accident or on purpose,
    and this row is meant to be evidence.
    """
    if not db.configured():
        raise HTTPException(
            status_code=503,
            detail="Supabase is not configured, so there is nothing to review and "
            "nowhere to write the decision. Set SUPABASE_URL and SUPABASE_SERVICE_KEY.",
        )

    entity = db.get_entity(req.entity_id)
    if entity is None:
        raise HTTPException(
            status_code=404, detail=f"No predicted entity with id {req.entity_id}."
        )

    row = db.insert_review(
        entity_id=req.entity_id,
        reviewer_id=req.reviewer_id,
        decision=req.decision,
        original_type=entity["entity_type"],
        original_start_char=entity["start_char"],
        original_end_char=entity["end_char"],
        original_confidence=entity["confidence"],
        corrected_type=req.corrected_type,
        corrected_start_char=req.corrected_start_char,
        corrected_end_char=req.corrected_end_char,
        note=req.note,
    )
    if row is None:
        raise HTTPException(
            status_code=502, detail="Supabase rejected the review insert. Check the API logs."
        )

    # The original prediction goes back to the caller untouched. If this ever
    # matches the correction, someone has started updating entities in place.
    return ReviewResponse(
        review_id=row["id"],
        entity_id=req.entity_id,
        decision=req.decision,
        reviewer_id=req.reviewer_id,
        original_type=entity["entity_type"],
        original_start_char=entity["start_char"],
        original_end_char=entity["end_char"],
        original_confidence=entity["confidence"],
        corrected_type=req.corrected_type,
        corrected_start_char=req.corrected_start_char,
        corrected_end_char=req.corrected_end_char,
        note=req.note,
        created_at=row.get("created_at"),
    )


@app.get("/runs", response_model=RunsResponse)
def runs(limit: int = 50) -> RunsResponse:
    """Training runs, newest first — the data behind the CRF vs. Transformer tab."""
    return RunsResponse(runs=[Run(**r) for r in db.latest_runs(limit=limit)])
