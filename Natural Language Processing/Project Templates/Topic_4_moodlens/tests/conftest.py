"""Shared pytest fixtures.

FINISHED CODE. The whole suite runs offline: the Supabase layer is replaced by
an in-memory fake, so you can develop and test the NLP layer with no cloud
account and no network. The tests that need real credentials or real compute
skip or are deselectable.

Markers (declared in pytest.ini):
    contract  — checks a function in api/nlp.py that you must implement.
                These FAIL until you write the code. That is the point: they
                are your to-do list. Run `pytest -m contract` to see what is left.
    network   — downloads a model or a corpus.
    train     — actually fine-tunes. Slow. `pytest -m "not train"` to skip.
    cloud     — needs real Supabase credentials; skipped otherwise.

The fakes below mirror api/db.py exactly, including the two-table split. If you
add a column to the schema, add it here too, or the tests will keep passing
against a shape the database no longer has.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture
def fake_db(monkeypatch):
    """Replace api.db with an in-memory store.

    Returns a dict with the two lists — ``runs`` and ``predictions`` — so a test
    can assert on what the route actually wrote.
    """
    from api import db

    store: dict[str, list[dict]] = {"runs": [], "predictions": []}

    def insert_run(
        model_version,
        base_model,
        config,
        metrics,
        dataset,
        n_train=None,
        n_eval=None,
        notes=None,
    ):
        row = {
            "id": len(store["runs"]) + 1,
            "model_version": model_version,
            "base_model": base_model,
            "dataset": dataset,
            "config": config,
            "metrics": metrics,
            "n_train": n_train,
            "n_eval": n_eval,
            "notes": notes,
            "created_at": "2026-01-01T00:00:00Z",
        }
        store["runs"].append(row)
        return row

    def insert_predictions(rows):
        inserted = []
        for row in rows:
            record = dict(row)
            record["id"] = len(store["predictions"]) + 1
            record.setdefault("created_at", "2026-01-01T00:00:00Z")
            store["predictions"].append(record)
            inserted.append(record)
        return inserted

    def insert_prediction(**kwargs):
        rows = insert_predictions([kwargs])
        return rows[0] if rows else None

    def latest_predictions(limit=100, label=None):
        rows = list(reversed(store["predictions"]))
        if label:
            rows = [r for r in rows if r["label"] == label]
        return rows[:limit]

    monkeypatch.setattr(db, "configured", lambda: True)
    monkeypatch.setattr(db, "ping", lambda: True)
    monkeypatch.setattr(db, "insert_run", insert_run)
    monkeypatch.setattr(db, "latest_runs", lambda limit=20: list(reversed(store["runs"]))[:limit])
    monkeypatch.setattr(db, "insert_prediction", insert_prediction)
    monkeypatch.setattr(db, "insert_predictions", insert_predictions)
    monkeypatch.setattr(db, "latest_predictions", latest_predictions)
    return store


@pytest.fixture
def client(fake_db):
    """A FastAPI TestClient with the database faked out."""
    from fastapi.testclient import TestClient

    from api.main import app

    return TestClient(app)


@pytest.fixture(scope="session")
def cloud_credentials():
    """Real Supabase credentials, or a skip."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not (url and key):
        pytest.skip("SUPABASE_URL / SUPABASE_SERVICE_KEY not set — skipping cloud test.")
    return url, key


@pytest.fixture
def prediction_factory():
    """Build SentimentPrediction objects without repeating six keyword arguments.

    Used by the evaluation contract tests, which need predictions but no model.
    """
    from shared.schemas import SentimentPrediction

    def make(probability: float, calibrated: bool = True, model_name: str = "test"):
        return SentimentPrediction(
            label="positive" if probability >= 0.5 else "negative",
            probability_positive=probability,
            raw_probability_positive=probability,
            confidence=max(probability, 1 - probability),
            calibrated=calibrated,
            model_name=model_name,
        )

    return make
