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
        "sequence_metadata": [],
    }
    counters = {"datasets": 0, "runs": 0}

    def insert_dataset(name, n_rows, seq_len, vocab_size, n_classes, seed):
        counters["datasets"] += 1
        row = {
            "id": counters["datasets"],
            "name": name,
            "n_rows": n_rows,
            "seq_len": seq_len,
            "vocab_size": vocab_size,
            "n_classes": n_classes,
            "seed": seed,
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        store["datasets"][row["id"]] = row
        return row

    def get_dataset(dataset_id):
        return store["datasets"].get(dataset_id)

    def _new_run(extra, model_b64):
        counters["runs"] += 1
        row = {
            "id": counters["runs"],
            "accuracy": None,
            "macro_f1": None,
            "scratch_accuracy": None,
            "val_loss": None,
            "pretrain_run_id": None,
            "created_at": "2026-01-01T00:00:00+00:00",
            **extra,
        }
        store["runs"][row["id"]] = row
        store["artifacts"][row["id"]] = model_b64
        return row

    def insert_pretrain_run(dataset_id, lr, batch_size, epochs, val_loss, model_b64):
        return _new_run(
            {
                "dataset_id": dataset_id,
                "run_type": "pretrain",
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "val_loss": val_loss,
            },
            model_b64,
        )

    def insert_finetune_run(
        dataset_id, pretrain_run_id, lr, batch_size, epochs, pretrained, scratch, model_b64
    ):
        return _new_run(
            {
                "dataset_id": dataset_id,
                "run_type": "finetune",
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "accuracy": pretrained["accuracy"] if pretrained else None,
                "macro_f1": pretrained["macro_f1"] if pretrained else None,
                "scratch_accuracy": scratch["accuracy"],
                "pretrain_run_id": pretrain_run_id,
            },
            model_b64,
        )

    def get_run(run_id):
        return store["runs"].get(run_id)

    def get_run_artifact(run_id):
        return store["artifacts"].get(run_id)

    def latest_runs(limit=50):
        return list(store["runs"].values())[-limit:][::-1]

    def insert_sequence_metadata(run_id, sha256, length, label, class_name, confidence):
        row = {
            "id": len(store["sequence_metadata"]) + 1,
            "run_id": run_id,
            "sha256": sha256,
            "length": length,
            "label": label,
            "class_name": class_name,
            "confidence": confidence,
        }
        store["sequence_metadata"].append(row)
        return row

    monkeypatch.setattr(db, "ping", lambda: True)
    monkeypatch.setattr(db, "insert_dataset", insert_dataset)
    monkeypatch.setattr(db, "get_dataset", get_dataset)
    monkeypatch.setattr(db, "insert_pretrain_run", insert_pretrain_run)
    monkeypatch.setattr(db, "insert_finetune_run", insert_finetune_run)
    monkeypatch.setattr(db, "get_run", get_run)
    monkeypatch.setattr(db, "get_run_artifact", get_run_artifact)
    monkeypatch.setattr(db, "latest_runs", latest_runs)
    monkeypatch.setattr(db, "insert_sequence_metadata", insert_sequence_metadata)

    from api.main import app

    with TestClient(app) as c:
        c._store = store  # expose for assertions
        yield c


def has_supabase_creds() -> bool:
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"))
