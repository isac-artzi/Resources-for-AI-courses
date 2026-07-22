"""Supabase persistence helpers for the API tier.

All Supabase access for the model service is funneled through this module. The
FastAPI app calls these functions; it never talks to Supabase directly. The
Streamlit UI NEVER imports this module -- it does its one read-only query with
the anon key on its own side.

The fitted model artifact is kept in a SEPARATE ``run_artifacts`` table (no anon
RLS policy) so the large base64 blob is never exposed to the public anon key --
only run metrics are anon-readable.

Environment variables (set locally in a .env, and in the Render dashboard):
    SUPABASE_URL              -> https://<project-ref>.supabase.co
    SUPABASE_SERVICE_KEY      -> the service-role key (server-side only, secret!)
"""
from __future__ import annotations

import os
from typing import List, Optional

from supabase import Client, create_client

_client: Optional[Client] = None

# Columns returned to callers as a "run" (metrics only, no model blob).
_RUN_COLS = (
    "id,dataset_id,hidden_dim,lr,batch_size,epochs,"
    "accuracy,precision,recall,f1,roc_auc,created_at"
)


def get_client() -> Client:
    """Lazily create and cache a Supabase client."""
    global _client
    if _client is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SERVICE_KEY"]
        _client = create_client(url, key)
    return _client


def ping() -> bool:
    """Return True if the Supabase client can reach the datasets table."""
    try:
        get_client().table("datasets").select("id").limit(1).execute()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
def insert_dataset(
    name: str,
    n_rows: int,
    n_features: int,
    positive_rate: float,
    records: List[dict],
    labels: List[int],
) -> dict:
    resp = (
        get_client()
        .table("datasets")
        .insert(
            {
                "name": name,
                "n_rows": n_rows,
                "n_features": n_features,
                "positive_rate": positive_rate,
                "records": records,
                "labels": labels,
            }
        )
        .execute()
    )
    return resp.data[0]


def get_dataset(dataset_id: int) -> Optional[dict]:
    resp = (
        get_client()
        .table("datasets")
        .select("*")
        .eq("id", dataset_id)
        .limit(1)
        .execute()
    )
    return resp.data[0] if resp.data else None


# ---------------------------------------------------------------------------
# runs (metrics)  +  run_artifacts (model blob)
# ---------------------------------------------------------------------------
def insert_run(
    dataset_id: int,
    hidden_dim: int,
    lr: float,
    batch_size: int,
    epochs: int,
    metrics: dict,
    model_b64: str,
) -> dict:
    client = get_client()
    resp = (
        client.table("runs")
        .insert(
            {
                "dataset_id": dataset_id,
                "hidden_dim": hidden_dim,
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            }
        )
        .execute()
    )
    run = resp.data[0]
    client.table("run_artifacts").insert(
        {"run_id": run["id"], "model_b64": model_b64}
    ).execute()
    return run


def get_run(run_id: int) -> Optional[dict]:
    resp = (
        get_client()
        .table("runs")
        .select(_RUN_COLS)
        .eq("id", run_id)
        .limit(1)
        .execute()
    )
    return resp.data[0] if resp.data else None


def get_run_artifact(run_id: int) -> Optional[str]:
    resp = (
        get_client()
        .table("run_artifacts")
        .select("model_b64")
        .eq("run_id", run_id)
        .limit(1)
        .execute()
    )
    return resp.data[0]["model_b64"] if resp.data else None


def latest_runs(limit: int = 50) -> List[dict]:
    resp = (
        get_client()
        .table("runs")
        .select(_RUN_COLS)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return resp.data


# ---------------------------------------------------------------------------
# predictions (the audit log)
# ---------------------------------------------------------------------------
def insert_prediction(run_id: int, features: dict, proba: float, label: int) -> dict:
    resp = (
        get_client()
        .table("predictions")
        .insert(
            {"run_id": run_id, "features": features, "proba": proba, "label": label}
        )
        .execute()
    )
    return resp.data[0]


def predictions_for_run(run_id: int) -> List[dict]:
    resp = (
        get_client()
        .table("predictions")
        .select("features,proba,label")
        .eq("run_id", run_id)
        .execute()
    )
    return resp.data
