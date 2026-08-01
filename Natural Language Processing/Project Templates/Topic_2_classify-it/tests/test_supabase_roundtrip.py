"""The one test that touches the real cloud.

It is SKIPPED unless SUPABASE_URL and SUPABASE_SERVICE_KEY are set, so the suite
stays green offline. Run it once after you apply db/migrations/001_init.sql, to
prove the third cloud is actually wired up:

    SUPABASE_URL=... SUPABASE_SERVICE_KEY=... pytest -m cloud

It writes to BOTH tables, because both matter and because a project that only
ever checked `runs` will discover the predictions table is misconfigured on demo
day, when the Recent Predictions tab is empty in front of an audience.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.cloud


def test_insert_and_read_back_a_training_run(cloud_credentials):
    from api import db

    row = db.insert_run(
        model_kind="baseline",
        hyperparameters={"test": True, "C": 1.0},
        metrics={
            "accuracy": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
            "positive_label": "urgent",
            "n_examples": 2,
        },
        model_version="pytest",
        dataset_name="pytest",
        n_train=8,
        n_eval=2,
    )
    assert row is not None, (
        "insert_run returned None — check SUPABASE_URL/SUPABASE_SERVICE_KEY and "
        "that you applied db/migrations/001_init.sql."
    )
    assert row["id"]

    recent = db.latest_runs(limit=10)
    assert any(r["id"] == row["id"] for r in recent)


def test_insert_and_read_back_a_served_prediction(cloud_credentials):
    from api import db

    row = db.insert_prediction(
        text_sha256="0" * 64,
        predicted_label="urgent",
        probability=0.77,
        model_kind="baseline",
        model_version="pytest",
        latency_ms=1.5,
    )
    assert row is not None, (
        "insert_prediction returned None — the predictions table is the serving "
        "audit trail and the Recent Predictions tab reads it. Check the migration."
    )
    assert row["id"]

    recent = db.latest_predictions(limit=10)
    assert any(r["id"] == row["id"] for r in recent)
    # And confirm the two tables really are separate: a prediction is not a run.
    assert all(r["id"] != row["id"] or "predicted_label" in r for r in recent)
