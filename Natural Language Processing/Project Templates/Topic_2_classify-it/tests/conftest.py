"""Shared pytest fixtures.

FINISHED CODE. The whole suite runs offline: the Supabase layer is replaced by
an in-memory fake, so you can develop and test the model layer with no cloud
account and no network. The one test that needs real credentials skips itself
when they are absent.

Markers (declared in pytest.ini):
    contract  — checks a function in api/nlp.py that you must implement.
                These FAIL until you write the code. That is the point: they
                are your to-do list. Run `pytest -m contract` to see what is left.
    model     — needs a trained artifact in artifacts/ (and the sklearn/torch
                stack installed). Slow. Deselect while you are working on the
                data layer: pytest -m "contract and not model"
    cloud     — needs real Supabase credentials; skipped otherwise.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture
def fake_db(monkeypatch):
    """Replace api.db with an in-memory store.

    Returns a dict with two lists, mirroring the two real tables:
        {"runs": [...], "predictions": [...]}
    Keeping them separate here is not decoration — a test that asserts a
    prediction was logged must not be satisfied by a training run.
    """
    from api import db

    store: dict[str, list[dict]] = {"runs": [], "predictions": []}

    def insert_run(
        model_kind,
        hyperparameters,
        metrics,
        model_version,
        dataset_name=None,
        n_train=None,
        n_eval=None,
    ):
        row = {
            "id": len(store["runs"]) + 1,
            "model_kind": model_kind,
            "dataset_name": dataset_name,
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "model_version": model_version,
            "n_train": n_train,
            "n_eval": n_eval,
            "created_at": "2026-01-01T00:00:00Z",
        }
        store["runs"].append(row)
        return row

    def insert_prediction(
        text_sha256,
        predicted_label,
        probability,
        model_kind,
        model_version,
        latency_ms=None,
    ):
        row = {
            "id": len(store["predictions"]) + 1,
            "text_sha256": text_sha256,
            "predicted_label": predicted_label,
            "probability": probability,
            "model_kind": model_kind,
            "model_version": model_version,
            "latency_ms": latency_ms,
            "created_at": "2026-01-01T00:00:00Z",
        }
        store["predictions"].append(row)
        return row

    def insert_predictions(rows):
        return [
            insert_prediction(
                text_sha256=r["text_sha256"],
                predicted_label=r["predicted_label"],
                probability=r["probability"],
                model_kind=r["model_kind"],
                model_version=r["model_version"],
                latency_ms=r.get("latency_ms"),
            )
            for r in rows
        ]

    monkeypatch.setattr(db, "configured", lambda: True)
    monkeypatch.setattr(db, "ping", lambda: True)
    monkeypatch.setattr(db, "insert_run", insert_run)
    monkeypatch.setattr(db, "insert_prediction", insert_prediction)
    monkeypatch.setattr(db, "insert_predictions", insert_predictions)
    monkeypatch.setattr(
        db, "latest_runs", lambda limit=50: list(reversed(store["runs"]))[:limit]
    )
    monkeypatch.setattr(
        db,
        "latest_predictions",
        lambda limit=100: list(reversed(store["predictions"]))[:limit],
    )
    return store


@pytest.fixture
def client(fake_db):
    """A FastAPI TestClient with the database faked out."""
    from fastapi.testclient import TestClient

    from api.main import app

    return TestClient(app)


@pytest.fixture
def tiny_corpus_csv(tmp_path):
    """A minimal two-label CSV on disk, for the loader contract test.

    Twelve rows, imbalanced 8/4, with a header and one blank-text row that your
    loader is expected to drop. Real corpora are messier than this; this is the
    floor, not the target.
    """
    path = tmp_path / "corpus.csv"
    path.write_text(
        "text,label\n"
        "my order never arrived,urgent\n"
        "i was charged twice,urgent\n"
        "still no response after four days,urgent\n"
        "this is unacceptable service,urgent\n"
        "please cancel immediately,urgent\n"
        "the app crashes on every launch,urgent\n"
        "refund me now,urgent\n"
        "nobody has replied,urgent\n"
        "how do i change my address,routine\n"
        "thanks that worked,routine\n"
        "where can i find the invoice,routine\n"
        ",routine\n",
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture(scope="session")
def cloud_credentials():
    """Real Supabase credentials, or a skip."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not (url and key):
        pytest.skip("SUPABASE_URL / SUPABASE_SERVICE_KEY not set — skipping cloud test.")
    return url, key
