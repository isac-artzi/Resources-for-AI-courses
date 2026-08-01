"""Supabase persistence for the API tier — Cloud #3.

FINISHED CODE. You should not need to change anything here to complete the
assignment, but you should read it, because it shows the rules the whole
three-cloud split depends on:

1. Only the API talks to Supabase with the SERVICE-ROLE key. The Streamlit tier
   never imports this module; it does its own read-only query with the ANON key.
2. Nothing here writes a raw prompt. Callers pass a sha256 hash. The generated
   OUTPUT is stored in full — that is the product, and the History tab has to
   show it — but the input the user typed is stored as a hash and nothing else.
3. Logging failures never fail the request. If Supabase is down the user still
   gets their text back, without a generation_id.

TWO TABLES, ON PURPOSE
----------------------
``training_runs``  — one row per fine-tuning run: base model, hyperparameters,
                     model version, corpus hash, held-out perplexity, timestamp.
``generations``    — one row per generated output: prompt hash, decoding
                     settings, the text, model version, timestamp, ratings.

They answer different questions. "Which learning rate produced v2?" is a
training_runs question. "What did the service say at 14:32 and how did the raters
score it?" is a generations question. A project that logs only training runs has
a History tab with nothing to display, because a fine-tune happens three times
and a generation happens three hundred.

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

    The API stays up without them so you can develop the generation layer
    offline; it skips logging and returns generation_id=None. /healthz reports
    the difference so you can tell "misconfigured" from "broken".
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
    """True if the generations table is reachable. Used by /healthz."""
    if not configured():
        return False
    try:
        get_client().table("generations").select("id").limit(1).execute()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# generations — one row per output the service produced
# ---------------------------------------------------------------------------
def insert_generation(
    prompt_sha256: str,
    strategy: str,
    decoding_params: Dict[str, Any],
    generated_text: str,
    model_version: str,
    prompt_token_count: Optional[int] = None,
    generated_token_count: Optional[int] = None,
    distinct_1: Optional[float] = None,
    distinct_2: Optional[float] = None,
    perplexity: Optional[float] = None,
    latency_ms: Optional[float] = None,
) -> Optional[dict]:
    """Log one generation. Returns the inserted row, or None.

    Logging must never take the request down with it: if Supabase is unreachable
    the user still gets their text back, just without a generation_id (and so
    without the ability to rate it). That is a deliberate design decision, not
    sloppiness — say so in your report if you are asked about failure modes.
    """
    if not configured():
        return None
    try:
        resp = (
            get_client()
            .table("generations")
            .insert(
                {
                    "prompt_sha256": prompt_sha256,
                    "strategy": strategy,
                    "decoding_params": decoding_params,
                    "generated_text": generated_text,
                    "model_version": model_version,
                    "prompt_token_count": prompt_token_count,
                    "generated_token_count": generated_token_count,
                    "distinct_1": distinct_1,
                    "distinct_2": distinct_2,
                    "perplexity": perplexity,
                    "latency_ms": latency_ms,
                }
            )
            .execute()
        )
        return resp.data[0] if resp.data else None
    except Exception:
        return None


def latest_generations(limit: int = 50) -> List[dict]:
    """Most recent generations, newest first. Backs GET /history and the UI tab."""
    if not configured():
        return []
    try:
        resp = (
            get_client()
            .table("generations")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception:
        return []


def get_generation(generation_id: int) -> Optional[dict]:
    """One generation by id, or None."""
    if not configured():
        return None
    try:
        resp = (
            get_client()
            .table("generations")
            .select("*")
            .eq("id", generation_id)
            .limit(1)
            .execute()
        )
        return resp.data[0] if resp.data else None
    except Exception:
        return None


def append_rating(generation_id: int, rating_payload: Dict[str, Any]) -> Optional[dict]:
    """Append one rater's record to a generation and refresh the aggregate.

    Every rating is kept, with its rater_id, in the ``ratings`` JSON array. The
    ``human_rating`` column holds the rounded mean of all of them, which is what
    the History table sorts and filters on. Keeping both is what lets you answer
    "how good was it?" and "did the two of you agree?" from the same row.

    Returns the updated row, or None if the generation does not exist or the
    write failed. A read-modify-write like this is fine at course scale; if two
    raters submitted in the same second you could lose one, and the honest fix is
    a separate ratings table with a foreign key — a reasonable thing to note in
    your report's limitations section.
    """
    if not configured():
        return None
    try:
        row = get_generation(generation_id)
        if row is None:
            return None

        ratings = list(row.get("ratings") or [])
        ratings.append(rating_payload)

        scores = [r["rating"] for r in ratings if isinstance(r.get("rating"), int)]
        aggregate = round(sum(scores) / len(scores)) if scores else None

        resp = (
            get_client()
            .table("generations")
            .update({"ratings": ratings, "human_rating": aggregate})
            .eq("id", generation_id)
            .execute()
        )
        return resp.data[0] if resp.data else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# training_runs — one row per fine-tuning run, written by your OFFLINE script
# ---------------------------------------------------------------------------
def insert_training_run(
    base_model: str,
    model_version: str,
    hyperparameters: Dict[str, Any],
    corpus_source: Optional[str] = None,
    corpus_sha256: Optional[str] = None,
    corpus_sentence_count: Optional[int] = None,
    held_out_perplexity: Optional[float] = None,
    notes: Optional[str] = None,
) -> Optional[dict]:
    """Record one fine-tuning run. Call this from your training script.

    There is no HTTP endpoint for this: training does not happen on the web
    service (the free instance cannot do it), so the row is written by whatever
    machine actually trained the model. Set SUPABASE_URL and
    SUPABASE_SERVICE_KEY in that environment and import this function.

    ``model_version`` is the join key. Stamp the same string on the service via
    the MODEL_VERSION environment variable, and every generation row can be
    traced back to the hyperparameters that produced its weights.
    """
    if not configured():
        return None
    try:
        resp = (
            get_client()
            .table("training_runs")
            .insert(
                {
                    "base_model": base_model,
                    "model_version": model_version,
                    "hyperparameters": hyperparameters,
                    "corpus_source": corpus_source,
                    "corpus_sha256": corpus_sha256,
                    "corpus_sentence_count": corpus_sentence_count,
                    "held_out_perplexity": held_out_perplexity,
                    "notes": notes,
                }
            )
            .execute()
        )
        return resp.data[0] if resp.data else None
    except Exception:
        return None


def latest_training_runs(limit: int = 20) -> List[dict]:
    """Most recent fine-tuning runs, newest first."""
    if not configured():
        return []
    try:
        resp = (
            get_client()
            .table("training_runs")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception:
        return []
