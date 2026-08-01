"""Supabase persistence for the API tier — Cloud #3.

FINISHED CODE. You should not need to change anything here to complete the
assignment, but you should read it, because it shows the three rules the whole
three-cloud split depends on:

1. Only the API talks to Supabase with the SERVICE-ROLE key. The Streamlit tier
   never imports this module; it does its own read-only query with the ANON key.
2. Nothing here writes raw user text. Callers pass a sha256 hash. If you find
   yourself wanting to store the message to make the Recent Predictions tab
   prettier, that is the moment to re-read the product brief.
3. TWO TABLES, TWO PURPOSES. ``runs`` records training: one row per time you
   fit a model, with its hyperparameters and its held-out metrics. Whereas
   ``predictions`` records serving: one row per question the deployed service
   answered. A service that logs its training but not its answers cannot tell a
   reviewer which model produced a given decision, and "we retrained since then"
   is not an audit trail. Keep them separate; do not be tempted to merge them
   because they both have a model_version column.

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

    The API stays up without them so you can develop the model layer offline; it
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
    """True if the runs table is reachable. Used by /healthz."""
    if not configured():
        return False
    try:
        get_client().table("runs").select("id").limit(1).execute()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# runs — one row per TRAINING run
# ---------------------------------------------------------------------------
def insert_run(
    model_kind: str,
    hyperparameters: Dict[str, Any],
    metrics: Dict[str, Any],
    model_version: str,
    dataset_name: Optional[str] = None,
    n_train: Optional[int] = None,
    n_eval: Optional[int] = None,
) -> Optional[dict]:
    """Log one training run. Returns the inserted row, or None.

    Called from your training script (see db/seed.py for the shape), not from a
    request handler — nothing in the web service trains a model.

    Store the hyperparameters that would change the result, all of them. Six
    months from now the difference between your two best runs will be one number
    in this JSON, and if it is the number you did not record, the run is gone.
    """
    if not configured():
        return None
    try:
        resp = (
            get_client()
            .table("runs")
            .insert(
                {
                    "model_kind": model_kind,
                    "dataset_name": dataset_name,
                    "hyperparameters": hyperparameters,
                    "metrics": metrics,
                    "model_version": model_version,
                    "n_train": n_train,
                    "n_eval": n_eval,
                }
            )
            .execute()
        )
        return resp.data[0] if resp.data else None
    except Exception:
        return None


def latest_runs(limit: int = 50) -> List[dict]:
    """Most recent training runs, newest first. Backs GET /runs and the
    Baseline vs. Transformer tab."""
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
# predictions — one row per SERVED prediction
# ---------------------------------------------------------------------------
def insert_prediction(
    text_sha256: str,
    predicted_label: str,
    probability: float,
    model_kind: str,
    model_version: str,
    latency_ms: Optional[float] = None,
) -> Optional[dict]:
    """Log one served prediction. Returns the inserted row, or None.

    Logging must never take the request down with it: if Supabase is unreachable
    the user still gets their classification back, just without a prediction_id.
    That is a deliberate design decision, not sloppiness — say so in your report
    if you are asked about failure modes.

    Note what is stored and what is not. The hash identifies the input without
    retaining it; ``model_version`` identifies which artifact answered, so a
    review months later can say "that decision came from distilbert-ft-v2, whose
    held-out recall was 0.71" by joining this table back to ``runs``. That join
    is the entire reason this is a separate table.
    """
    if not configured():
        return None
    try:
        resp = (
            get_client()
            .table("predictions")
            .insert(
                {
                    "text_sha256": text_sha256,
                    "predicted_label": predicted_label,
                    "probability": probability,
                    "model_kind": model_kind,
                    "model_version": model_version,
                    "latency_ms": latency_ms,
                }
            )
            .execute()
        )
        return resp.data[0] if resp.data else None
    except Exception:
        return None


def insert_predictions(rows: List[Dict[str, Any]]) -> List[dict]:
    """Bulk-insert served predictions (used by /predict_batch).

    One round trip for the whole batch. Inserting 64 rows one at a time turns a
    fast endpoint into 64 sequential network calls, which on the free plan is the
    slowest thing in the request by a wide margin.
    """
    if not configured() or not rows:
        return []
    try:
        resp = get_client().table("predictions").insert(rows).execute()
        return resp.data or []
    except Exception:
        return []


def latest_predictions(limit: int = 100) -> List[dict]:
    """Most recent served predictions, newest first. Backs the Recent
    Predictions tab (which reads this table directly with the anon key)."""
    if not configured():
        return []
    try:
        resp = (
            get_client()
            .table("predictions")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception:
        return []
