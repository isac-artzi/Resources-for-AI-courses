"""Supabase persistence helpers for the API tier.

All SQL/Supabase access for the model service is funneled through this module.
The FastAPI app calls these functions; it never talks to Supabase directly. The
Streamlit UI NEVER imports this module -- it does its one read-only query with
the anon key on its own side.

Environment variables (set locally in a .env, and in the Render dashboard):
    SUPABASE_URL              -> https://<project-ref>.supabase.co
    SUPABASE_SERVICE_KEY      -> the service-role key (server-side only, secret!)
"""
from __future__ import annotations

import os
from typing import List, Optional

from supabase import Client, create_client

_client: Optional[Client] = None


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
    slope: float,
    intercept: float,
    noise: float,
    n_points: int,
    xs: List[float],
    ys: List[float],
) -> dict:
    resp = (
        get_client()
        .table("datasets")
        .insert(
            {
                "name": name,
                "slope": slope,
                "intercept": intercept,
                "noise": noise,
                "n_points": n_points,
                "xs": xs,
                "ys": ys,
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
# runs
# ---------------------------------------------------------------------------
def insert_run(
    dataset_id: int,
    lr: float,
    batch_size: int,
    epochs: int,
    mse: float,
    mae: float,
    r2: float,
    weights_json: dict,
) -> dict:
    resp = (
        get_client()
        .table("runs")
        .insert(
            {
                "dataset_id": dataset_id,
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "weights_json": weights_json,
            }
        )
        .execute()
    )
    return resp.data[0]


def get_run(run_id: int) -> Optional[dict]:
    resp = get_client().table("runs").select("*").eq("id", run_id).limit(1).execute()
    return resp.data[0] if resp.data else None


def latest_runs(limit: int = 50) -> List[dict]:
    resp = (
        get_client()
        .table("runs")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return resp.data


# ---------------------------------------------------------------------------
# predictions
# ---------------------------------------------------------------------------
def insert_prediction(run_id: int, x: float, yhat: float) -> dict:
    resp = (
        get_client()
        .table("predictions")
        .insert({"run_id": run_id, "x": x, "yhat": yhat})
        .execute()
    )
    return resp.data[0]
