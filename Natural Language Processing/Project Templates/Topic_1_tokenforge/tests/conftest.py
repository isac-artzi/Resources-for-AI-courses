"""Shared pytest fixtures.

FINISHED CODE. The whole suite runs offline: the Supabase layer is replaced by
an in-memory fake, so you can develop and test the NLP layer with no cloud
account and no network. The one test that needs real credentials skips itself
when they are absent.

Markers (declared in pytest.ini):
    contract  — checks a function in api/nlp.py that you must implement.
                These FAIL until you write the code. That is the point: they
                are your to-do list. Run `pytest -m contract` to see what is left.
    cloud     — needs real Supabase credentials; skipped otherwise.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture
def fake_db(monkeypatch):
    """Replace api.db with an in-memory store. Returns the store for assertions."""
    from api import db

    store: list[dict] = []

    def insert_run(
        kind,
        text_sha256,
        config,
        model_version,
        token_count_before=None,
        token_count_after=None,
        oov_rate=None,
    ):
        row = {
            "id": len(store) + 1,
            "kind": kind,
            "text_sha256": text_sha256,
            "config": config,
            "model_version": model_version,
            "token_count_before": token_count_before,
            "token_count_after": token_count_after,
            "oov_rate": oov_rate,
            "created_at": "2026-01-01T00:00:00Z",
        }
        store.append(row)
        return row

    monkeypatch.setattr(db, "configured", lambda: True)
    monkeypatch.setattr(db, "ping", lambda: True)
    monkeypatch.setattr(db, "insert_run", insert_run)
    monkeypatch.setattr(db, "latest_runs", lambda limit=50: list(reversed(store))[:limit])
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
