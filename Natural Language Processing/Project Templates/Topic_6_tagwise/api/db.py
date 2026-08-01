"""Supabase persistence for the API tier — Cloud #3.

FINISHED CODE. You should not need to change anything here to complete the
assignment, but you should read it, because it shows the two rules the whole
three-cloud split depends on:

1. Only the API talks to Supabase with the SERVICE-ROLE key. The Streamlit tier
   never imports this module; it does its own read-only query with the ANON key.
2. Nothing here writes raw user text. Callers pass a sha256 hash. If you find
   yourself wanting to store the sentence to make the History tab prettier, that
   is the moment to re-read the product brief.

TWO TABLES, ON PURPOSE
----------------------
``runs``     — one row per tagger build (lookup table built, transformer
               fine-tuned): hyperparameters, metrics, model version.
``taggings`` — one row per served request: hashed sentence, predicted tag
               sequence, which model answered, timestamp.

They answer different questions. ``runs`` tells you how good a tagger was on a
held-out split at build time. ``taggings`` tells you what the service actually
did in production — which words it saw, how often the baseline fell back, how
traffic shifted after you shipped a new model version. Log only ``runs`` and the
History tab has no data source: a build log is not a request log.

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
    skips logging and returns tagging_id=None. /healthz reports the difference so
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
    """True if both tables are reachable. Used by /healthz.

    It checks ``taggings`` as well as ``runs`` on purpose: applying half the
    migration is a real failure mode, and it shows up as an empty History tab
    hours later instead of as a red health check now.
    """
    if not configured():
        return False
    try:
        get_client().table("runs").select("id").limit(1).execute()
        get_client().table("taggings").select("id").limit(1).execute()
        return True
    except Exception:
        return False


def insert_run(
    model: str,
    tagset: str,
    hyperparameters: Dict[str, Any],
    model_version: str,
    accuracy: Optional[float] = None,
    macro_f1: Optional[float] = None,
    metrics: Optional[Dict[str, Any]] = None,
    notes: Optional[str] = None,
) -> Optional[dict]:
    """Record one tagger build. Returns the inserted row, or None.

    Call this from your training/build script (not from a request handler — a
    web request must never fine-tune a model). ``metrics`` is where the confusion
    matrix goes, as {"confusion": {"labels": [...], "matrix": [[...]]}}, because
    the comparison tab renders it straight out of this row.
    """
    if not configured():
        return None
    try:
        resp = (
            get_client()
            .table("runs")
            .insert(
                {
                    "model": model,
                    "tagset": tagset,
                    "hyperparameters": hyperparameters,
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "metrics": metrics or {},
                    "model_version": model_version,
                    "notes": notes,
                }
            )
            .execute()
        )
        return resp.data[0] if resp.data else None
    except Exception:
        return None


def insert_tagging(
    sentence_sha256: str,
    token_count: int,
    tag_sequence: List[str],
    model: str,
    model_version: str,
    unknown_count: int = 0,
) -> Optional[dict]:
    """Log one served tagging request. Returns the inserted row, or None.

    Logging must never take the request down with it: if Supabase is unreachable
    the user still gets their tags back, without a tagging_id. That is a
    deliberate design decision, not sloppiness — say so in your report if you are
    asked about failure modes.
    """
    if not configured():
        return None
    try:
        resp = (
            get_client()
            .table("taggings")
            .insert(
                {
                    "sentence_sha256": sentence_sha256,
                    "token_count": token_count,
                    "tag_sequence": tag_sequence,
                    "model": model,
                    "model_version": model_version,
                    "unknown_count": unknown_count,
                }
            )
            .execute()
        )
        return resp.data[0] if resp.data else None
    except Exception:
        return None


def latest_runs(limit: int = 50) -> List[dict]:
    """Most recent builds, newest first. Backs GET /runs and the comparison tab."""
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


def latest_taggings(limit: int = 100) -> List[dict]:
    """Most recent served requests, newest first.

    The UI reads the same rows directly with the anon key, so nothing calls this
    today. It exists so your own scripts and tests can read the request log
    without duplicating the query.
    """
    if not configured():
        return []
    try:
        resp = (
            get_client()
            .table("taggings")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception:
        return []
