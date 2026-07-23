"""Shared pytest fixtures.

The schema, health, and numerical tests run WITHOUT any cloud access by stubbing
the db module with an in-memory store. The Supabase round-trip test is skipped
unless real credentials are present, so `pytest` stays green in CI/offline.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture
def client(monkeypatch):
    """A TestClient with the Supabase layer stubbed by an in-memory fake."""
    from fastapi.testclient import TestClient

    from api import db

    store = {
        "datasets": {},
        "runs": {},
        "artifacts": {},
        "image_metadata": [],
    }
    counters = {"datasets": 0, "runs": 0}

    def insert_dataset(name, n_rows, img_size, n_classes, noise, seed):
        counters["datasets"] += 1
        row = {
            "id": counters["datasets"],
            "name": name,
            "n_rows": n_rows,
            "img_size": img_size,
            "n_classes": n_classes,
            "noise": noise,
            "seed": seed,
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        store["datasets"][row["id"]] = row
        return row

    def get_dataset(dataset_id):
        return store["datasets"].get(dataset_id)

    def insert_run(dataset_id, lr, batch_size, epochs, metrics, model_b64):
        counters["runs"] += 1
        row = {
            "id": counters["runs"],
            "dataset_id": dataset_id,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "recon_loss": metrics["recon_loss"],
            "kl": metrics["kl"],
            "elbo": metrics["elbo"],
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        store["runs"][row["id"]] = row
        store["artifacts"][row["id"]] = model_b64
        return row

    def get_run(run_id):
        return store["runs"].get(run_id)

    def get_run_artifact(run_id):
        return store["artifacts"].get(run_id)

    def latest_runs(limit=50):
        return list(store["runs"].values())[-limit:][::-1]

    def insert_image_metadata(run_id, sha256, width, height, recon_mse):
        row = {
            "id": len(store["image_metadata"]) + 1,
            "run_id": run_id,
            "sha256": sha256,
            "width": width,
            "height": height,
            "recon_mse": recon_mse,
        }
        store["image_metadata"].append(row)
        return row

    monkeypatch.setattr(db, "ping", lambda: True)
    monkeypatch.setattr(db, "insert_dataset", insert_dataset)
    monkeypatch.setattr(db, "get_dataset", get_dataset)
    monkeypatch.setattr(db, "insert_run", insert_run)
    monkeypatch.setattr(db, "get_run", get_run)
    monkeypatch.setattr(db, "get_run_artifact", get_run_artifact)
    monkeypatch.setattr(db, "latest_runs", latest_runs)
    monkeypatch.setattr(db, "insert_image_metadata", insert_image_metadata)

    from api.main import app

    with TestClient(app) as c:
        c._store = store  # expose for assertions
        yield c


def has_supabase_creds() -> bool:
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"))
