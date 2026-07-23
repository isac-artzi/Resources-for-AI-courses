"""Supabase round-trip test.

Registers a dataset, pretrains, fine-tunes against it, and confirms the run rows
were written. Skipped automatically unless real Supabase credentials are present,
so the suite still passes offline. Run it against a real (throwaway) project with:

    SUPABASE_URL=... SUPABASE_SERVICE_KEY=... pytest tests/test_supabase_roundtrip.py
"""
from __future__ import annotations

import pytest

from tests.conftest import has_supabase_creds

pytestmark = pytest.mark.skipif(
    not has_supabase_creds(), reason="No live Supabase credentials in environment."
)


def test_dataset_pretrain_finetune_roundtrip():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as c:
        ds = c.post(
            "/datasets", json={"name": "roundtrip", "n_rows": 1000, "seq_len": 20}
        ).json()

        pre = c.post(
            "/pretrain",
            json={"dataset_id": ds["id"], "lr": 0.005, "batch_size": 64, "epochs": 4},
        )
        assert pre.status_code == 200
        pretrain_run_id = pre.json()["run_id"]

        ft = c.post(
            "/finetune",
            json={"dataset_id": ds["id"], "pretrain_run_id": pretrain_run_id, "epochs": 4},
        )
        assert ft.status_code == 200
        run_id = ft.json()["run_id"]

        got = c.get(f"/runs/{run_id}")
        assert got.status_code == 200
        assert got.json()["dataset_id"] == ds["id"]
        assert got.json()["run_type"] == "finetune"
