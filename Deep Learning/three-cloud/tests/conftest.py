"""Shared pytest fixtures.

The schema, health, and numerical tests run WITHOUT any cloud access by stubbing
the db module. The Supabase round-trip test is skipped unless real credentials
are present in the environment, so `pytest` stays green in CI/offline.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture
def client(monkeypatch):
    """A TestClient with the Supabase layer stubbed by an in-memory fake."""
    from fastapi.testclient import TestClient

    from api import db

    store = {"datasets": {}, "runs": {}, "predictions": []}
    counters = {"datasets": 0, "runs": 0}

    def insert_dataset(name, slope, intercept, noise, n_points, xs, ys):
        counters["datasets"] += 1
        row = {
            "id": counters["datasets"],
            "name": name,
            "slope": slope,
            "intercept": intercept,
            "noise": noise,
            "n_points": n_points,
            "xs": xs,
            "ys": ys,
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        store["datasets"][row["id"]] = row
        return row

    def get_dataset(dataset_id):
        return store["datasets"].get(dataset_id)

    def insert_run(dataset_id, lr, batch_size, epochs, mse, mae, r2, weights_json):
        counters["runs"] += 1
        row = {
            "id": counters["runs"],
            "dataset_id": dataset_id,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "weights_json": weights_json,
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        store["runs"][row["id"]] = row
        return row

    def get_run(run_id):
        return store["runs"].get(run_id)

    def latest_runs(limit=50):
        return list(store["runs"].values())[-limit:][::-1]

    def insert_prediction(run_id, x, yhat):
        row = {"id": len(store["predictions"]) + 1, "run_id": run_id, "x": x, "yhat": yhat}
        store["predictions"].append(row)
        return row

    monkeypatch.setattr(db, "ping", lambda: True)
    monkeypatch.setattr(db, "insert_dataset", insert_dataset)
    monkeypatch.setattr(db, "get_dataset", get_dataset)
    monkeypatch.setattr(db, "insert_run", insert_run)
    monkeypatch.setattr(db, "get_run", get_run)
    monkeypatch.setattr(db, "latest_runs", latest_runs)
    monkeypatch.setattr(db, "insert_prediction", insert_prediction)

    from api.main import app

    with TestClient(app) as c:
        c._store = store  # expose for assertions
        yield c


def has_supabase_creds() -> bool:
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"))
