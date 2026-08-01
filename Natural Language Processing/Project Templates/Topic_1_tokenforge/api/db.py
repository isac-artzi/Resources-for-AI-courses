"""Supabase persistence for the API tier — Cloud #3.

FINISHED CODE. You should not need to change anything here to complete the
assignment, but you should read it, because it shows the two rules the whole
three-cloud split depends on:

1. Only the API talks to Supabase with the SERVICE-ROLE key. The Streamlit tier
   never imports this module; it does its own read-only query with the ANON key.
2. Nothing here writes raw user text. Callers pass a sha256 hash. If you find
   yourself wanting to store the text to make the History tab prettier, that is
   the moment to re-read the product brief.

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
    just skips logging and returns run_id=None. /healthz reports the difference
    so you can tell "misconfigured" from "broken".
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


def insert_run(
    kind: str,
    text_sha256: str,
    config: Dict[str, Any],
    model_version: str,
    token_count_before: Optional[int] = None,
    token_count_after: Optional[int] = None,
    oov_rate: Optional[float] = None,
) -> Optional[dict]:
    """Log one pass through the service. Returns the inserted row, or None.

    Logging must never take the request down with it: if Supabase is unreachable
    the user still gets their tokenization back, just without a run_id. That is
    a deliberate design decision, not sloppiness — say so in your report if you
    are asked about failure modes.
    """
    if not configured():
        return None
    try:
        resp = (
            get_client()
            .table("runs")
            .insert(
                {
                    "kind": kind,
                    "text_sha256": text_sha256,
                    "config": config,
                    "model_version": model_version,
                    "token_count_before": token_count_before,
                    "token_count_after": token_count_after,
                    "oov_rate": oov_rate,
                }
            )
            .execute()
        )
        return resp.data[0] if resp.data else None
    except Exception:
        return None


def latest_runs(limit: int = 50) -> List[dict]:
    """Most recent runs, newest first. Backs GET /runs and the History tab."""
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
