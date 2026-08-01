"""Supabase persistence for the API tier — Cloud #3.

FINISHED CODE. You should not need to change anything here to complete the
assignment, but you should read it, because it shows the three rules the whole
three-cloud split depends on:

1. Only the API talks to Supabase with the SERVICE-ROLE key. The Streamlit tier
   never imports this module; it does its own read-only queries with the ANON
   key.
2. Nothing here writes raw review text. Callers pass a sha256 hash. If you find
   yourself wanting to store the text to make the audit view prettier, that is
   the moment to re-read the product brief.
3. Logging never takes the request down with it. A prediction still reaches the
   analyst when Supabase is unreachable; it just comes back without an id.

TWO TABLES, TWO JOBS
--------------------
``insert_run``        — one row per TRAINING run. Hyperparameters and metrics.
                        Answers "how good is the model, and how do we know?"
``insert_prediction`` — one row per SERVED prediction. Answers "what did we tell
                        this user, when, and from which build?"

Neither substitutes for the other. Metrics on a held-out split say nothing about
what the deployed service actually did last Tuesday, and a log of predictions
with no evaluation behind it says nothing about whether any of them were right.
The audit story the product brief asks for needs both rows to exist and to be
joinable by ``model_version``.

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
    just skips logging and returns prediction_id=None. /healthz reports the
    difference so you can tell "misconfigured" from "broken".
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
    """True if the predictions table is reachable. Used by /healthz."""
    if not configured():
        return False
    try:
        get_client().table("predictions").select("id").limit(1).execute()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# runs — training runs
# ---------------------------------------------------------------------------
def insert_run(
    model_version: str,
    base_model: str,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    dataset: str,
    n_train: Optional[int] = None,
    n_eval: Optional[int] = None,
    notes: Optional[str] = None,
) -> Optional[dict]:
    """Log one training run and its evaluation. Returns the row, or None.

    ``metrics`` is the whole evaluation payload as JSON — document metrics for
    both models, per-aspect metrics, and the slice breakdown. The Model
    Performance and Bias Audit tabs read this column directly, so its shape is
    part of the contract: see ``DocumentMetrics``, ``AspectMetrics`` and
    ``SliceMetrics`` in shared/schemas.py, and the seed row in db/seed.py for a
    worked example of the layout.

    Call this from your training script, not from a request handler. Training
    does not happen on Render.
    """
    if not configured():
        return None
    try:
        resp = (
            get_client()
            .table("runs")
            .insert(
                {
                    "model_version": model_version,
                    "base_model": base_model,
                    "dataset": dataset,
                    "config": config,
                    "metrics": metrics,
                    "n_train": n_train,
                    "n_eval": n_eval,
                    "notes": notes,
                }
            )
            .execute()
        )
        return resp.data[0] if resp.data else None
    except Exception:
        return None


def latest_runs(limit: int = 20) -> List[dict]:
    """Most recent training runs, newest first."""
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
# predictions — everything the service served
# ---------------------------------------------------------------------------
def insert_prediction(
    text_sha256: str,
    label: str,
    probability_positive: float,
    confidence: float,
    calibrated: bool,
    aspects: List[Dict[str, Any]],
    model_name: str,
    model_version: str,
    char_count: Optional[int] = None,
) -> Optional[dict]:
    """Log one served prediction, aspect breakdown included. Returns the row.

    The aspect breakdown is stored WITH the prediction rather than in a side
    table on purpose: an audit that can see the label but not which aspect drove
    it cannot answer the only question an audit is ever asked, which is why.
    """
    rows = insert_predictions(
        [
            {
                "text_sha256": text_sha256,
                "char_count": char_count,
                "label": label,
                "probability_positive": probability_positive,
                "confidence": confidence,
                "calibrated": calibrated,
                "aspects": aspects,
                "model_name": model_name,
                "model_version": model_version,
            }
        ]
    )
    return rows[0] if rows else None


def insert_predictions(rows: List[Dict[str, Any]]) -> List[dict]:
    """Insert several predictions in ONE round trip. Returns the inserted rows.

    /predict_batch uses this. Inserting sixty rows one at a time means sixty
    HTTPS round trips to Supabase, which will dominate the batch's runtime and
    make you blame the model for being slow.
    """
    if not configured() or not rows:
        return []
    try:
        resp = get_client().table("predictions").insert(rows).execute()
        return resp.data or []
    except Exception:
        return []


def latest_predictions(limit: int = 100, label: Optional[str] = None) -> List[dict]:
    """Most recent predictions, newest first. Backs GET /audit."""
    if not configured():
        return []
    try:
        query = get_client().table("predictions").select("*")
        if label:
            query = query.eq("label", label)
        resp = query.order("created_at", desc=True).limit(limit).execute()
        return resp.data or []
    except Exception:
        return []
