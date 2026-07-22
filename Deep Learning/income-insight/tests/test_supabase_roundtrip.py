"""Supabase round-trip test.

Inserts a fixture dataset, trains against it, and confirms a run row was written.
Skipped automatically unless real Supabase credentials are present, so the suite
still passes offline. Run it against a real (throwaway) project with:

    SUPABASE_URL=... SUPABASE_SERVICE_KEY=... pytest tests/test_supabase_roundtrip.py
"""
from __future__ import annotations

import pytest

from tests.conftest import has_supabase_creds

pytestmark = pytest.mark.skipif(
    not has_supabase_creds(), reason="No live Supabase credentials in environment."
)


def test_dataset_train_run_roundtrip():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as c:
        ds = c.post(
            "/datasets", json={"name": "roundtrip", "n_rows": 500, "noise": 1.0}
        ).json()
        resp = c.post(
            "/train",
            json={"dataset_id": ds["id"], "hidden_dim": 16, "lr": 0.01,
                  "batch_size": 32, "epochs": 50},
        )
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        got = c.get(f"/runs/{run_id}")
        assert got.status_code == 200
        assert got.json()["dataset_id"] == ds["id"]
