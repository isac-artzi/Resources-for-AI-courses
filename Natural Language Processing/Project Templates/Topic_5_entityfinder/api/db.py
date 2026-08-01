"""Supabase persistence for the API tier — Cloud #3.

FINISHED CODE. You should not need to change anything here to complete the
assignment, but you should read it, because it shows the rules the whole
three-cloud split depends on:

1. Only the API talks to Supabase with the SERVICE-ROLE key. The Streamlit tier
   never imports this module; its one direct read uses the ANON key, and every
   write goes through an HTTP endpoint here.
2. Nothing here writes a raw document. Callers pass a sha256 hash. The entity
   surface strings are stored — they are the predictions, and a review queue
   without them is unusable — but the document they came from is not.
3. A review NEVER updates the entities row. It inserts into ``reviews``. See
   ``insert_review`` for why.

Environment variables (a .env locally, the Render dashboard in production):
    SUPABASE_URL          -> https://<project-ref>.supabase.co
    SUPABASE_SERVICE_KEY  -> service-role key. Secret. Never in the UI tier.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from supabase import Client, create_client

_client: Optional[Client] = None


def configured() -> bool:
    """True when both Supabase environment variables are present.

    The API stays up without them so you can develop the NLP layer offline; it
    just skips logging and returns null ids. /healthz reports the difference so
    you can tell "misconfigured" from "broken".
    """
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"))


def get_client() -> Client:
    """Lazily create and cache the Supabase client."""
    global _client
    if _client is None:
        _client = create_client(
            os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"]
        )
    return _client


def ping() -> bool:
    """True if the runs table is reachable. Used by /healthz."""
    if not configured():
        return False
    try:
        get_client().table("runs").select("id").limit(1).execute()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Training runs
# ---------------------------------------------------------------------------
def insert_run(
    model_type: str,
    dataset: str,
    config: Dict[str, Any],
    model_version: str,
    precision: Optional[float] = None,
    recall: Optional[float] = None,
    f1: Optional[float] = None,
    metrics: Optional[Dict[str, Any]] = None,
    notes: Optional[str] = None,
) -> Optional[dict]:
    """Log one training run — one row per model, per configuration.

    Both models write here, which is what makes the CRF vs. Transformer tab a
    single query instead of a pair of pasted screenshots. ``config`` holds the
    hyperparameters and, for the CRF, the feature list; ``metrics`` holds the
    full entity-level scores including the per-type breakdown.

    Logging must never take the request down with it — see insert_extraction.
    """
    if not configured():
        return None
    try:
        resp = (
            get_client()
            .table("runs")
            .insert(
                {
                    "model_type": model_type,
                    "dataset": dataset,
                    "config": config,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "metrics": metrics,
                    "model_version": model_version,
                    "notes": notes,
                }
            )
            .execute()
        )
        return resp.data[0] if resp.data else None
    except Exception:
        return None


def latest_runs(limit: int = 50) -> List[dict]:
    """Most recent training runs, newest first. Backs GET /runs."""
    if not configured():
        return []
    try:
        resp = (
            get_client()
            .table("runs")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Served extractions and the entities they produced
# ---------------------------------------------------------------------------
def insert_extraction(
    text_sha256: str,
    model: str,
    model_version: str,
    entity_count: int,
    latency_ms: Optional[int] = None,
) -> Optional[dict]:
    """Log one served extraction. Returns the inserted row, or None.

    Logging must never take the request down with it: if Supabase is unreachable
    the analyst still gets their entities back, just without ids and without a
    row in the review queue. That is a deliberate design decision, not
    sloppiness — say so in your report if you are asked about failure modes.
    """
    if not configured():
        return None
    try:
        resp = (
            get_client()
            .table("extractions")
            .insert(
                {
                    "text_sha256": text_sha256,
                    "model": model,
                    "model_version": model_version,
                    "entity_count": entity_count,
                    "latency_ms": latency_ms,
                }
            )
            .execute()
        )
        return resp.data[0] if resp.data else None
    except Exception:
        return None


def insert_entities(extraction_id: int, entities: List[Dict[str, Any]]) -> List[dict]:
    """Log every predicted entity for one extraction. Returns the inserted rows.

    One INSERT with a list, not one per entity — a document with forty entities
    should not cost forty round trips.

    The returned ids matter: they go back to the UI in the extract response and
    they are what ``POST /review`` addresses. An entity the database never saw
    cannot be reviewed.
    """
    if not configured() or not entities:
        return []
    payload = [
        {
            "extraction_id": extraction_id,
            "text": e["text"],
            "start_char": e["start_char"],
            "end_char": e["end_char"],
            "entity_type": e["entity_type"],
            "confidence": e["confidence"],
            "context": e.get("context"),
        }
        for e in entities
    ]
    try:
        resp = get_client().table("entities").insert(payload).execute()
        return resp.data or []
    except Exception:
        return []


def get_entity(entity_id: int) -> Optional[dict]:
    """Fetch one predicted entity by id, with its parent extraction's model info.

    POST /review calls this before it writes, so the review row can carry a
    snapshot of what the model originally said.
    """
    if not configured():
        return None
    try:
        resp = (
            get_client()
            .table("entities")
            .select("*, extractions(model, model_version, text_sha256)")
            .eq("id", entity_id)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        return rows[0] if rows else None
    except Exception:
        return None


def review_queue(
    threshold: float,
    limit: int = 50,
    include_reviewed: bool = False,
) -> List[dict]:
    """Predicted entities scoring below ``threshold``, lowest confidence first.

    Already-reviewed entities are filtered out by default, because a queue that
    hands a reviewer the same row after they have ruled on it is a queue nobody
    uses twice. Two queries rather than a join: PostgREST will not give us a
    left-anti-join, and the reviewed set is small.
    """
    if not configured():
        return []
    try:
        client = get_client()
        resp = (
            client.table("entities")
            .select("*, extractions(model, model_version)")
            .lt("confidence", threshold)
            .order("confidence", desc=False)
            .limit(limit * 2 if not include_reviewed else limit)
            .execute()
        )
        rows = resp.data or []
        if not include_reviewed and rows:
            reviewed = (
                client.table("reviews")
                .select("entity_id")
                .in_("entity_id", [r["id"] for r in rows])
                .execute()
            )
            done = {r["entity_id"] for r in (reviewed.data or [])}
            rows = [r for r in rows if r["id"] not in done]
        return rows[:limit]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# The write path that makes the review queue a review queue
# ---------------------------------------------------------------------------
def insert_review(
    entity_id: int,
    reviewer_id: str,
    decision: str,
    original_type: str,
    original_start_char: int,
    original_end_char: int,
    original_confidence: float,
    corrected_type: Optional[str] = None,
    corrected_start_char: Optional[int] = None,
    corrected_end_char: Optional[int] = None,
    note: Optional[str] = None,
) -> Optional[dict]:
    """Record a reviewer's decision as a NEW ROW. Never an UPDATE. Ever.

    It is tempting to write the correction straight onto the entities row: one
    table, always current, no joins. Do not. Two things die when you do.

    The audit trail dies first. Once the row says ORG, nobody can answer "did the
    model get this right, or did a human fix it?" — and that question is the
    entire reason the product logs anything. Your error analysis, your
    per-version comparison, and any claim you make about the model's quality all
    depend on the prediction still being there to compare against.

    The training signal dies second, and more expensively. A corrected entity is
    a labelled example of a mistake this model makes: the pair (what it said,
    what was true) is exactly what you would fine-tune on next. Overwrite the
    prediction and you keep the label but throw away the error — you are left
    with ordinary annotation instead of the hard cases your reviewers were
    already paying attention to.

    So: entities is append-only and immutable, reviews is append-only, and the
    current best answer for an entity is "its most recent review, or the
    prediction if it has none". That is a view, not a mutation.
    """
    if not configured():
        return None
    try:
        resp = (
            get_client()
            .table("reviews")
            .insert(
                {
                    "entity_id": entity_id,
                    "reviewer_id": reviewer_id,
                    "decision": decision,
                    "original_type": original_type,
                    "original_start_char": original_start_char,
                    "original_end_char": original_end_char,
                    "original_confidence": original_confidence,
                    "corrected_type": corrected_type,
                    "corrected_start_char": corrected_start_char,
                    "corrected_end_char": corrected_end_char,
                    "note": note,
                }
            )
            .execute()
        )
        return resp.data[0] if resp.data else None
    except Exception:
        return None


def latest_reviews(limit: int = 50) -> List[dict]:
    """Most recent reviewer decisions, newest first. Useful for the report:
    the accept/correct/reject mix is your error breakdown, already counted."""
    if not configured():
        return []
    try:
        resp = (
            get_client()
            .table("reviews")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception:
        return []
