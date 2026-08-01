"""The one test that touches the real cloud.

It is SKIPPED unless SUPABASE_URL and SUPABASE_SERVICE_KEY are set, so the suite
stays green offline. Run it once after you apply db/migrations/001_init.sql, to
prove the third cloud is actually wired up:

    SUPABASE_URL=... SUPABASE_SERVICE_KEY=... pytest -m cloud

It writes to BOTH tables, because both are part of the audit story and a schema
that only half applied is a problem you want to find now.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.cloud


def test_insert_and_read_back_a_training_run(cloud_credentials):
    from api import db

    row = db.insert_run(
        model_version="pytest",
        base_model="none",
        dataset="pytest fixture",
        config={"test": True},
        metrics={"documents": [], "aspects": [], "slices": []},
        n_train=0,
        n_eval=0,
        notes="written by the cloud round-trip test",
    )
    assert row is not None, (
        "insert_run returned None — check SUPABASE_URL/SUPABASE_SERVICE_KEY and "
        "that you applied db/migrations/001_init.sql."
    )
    assert row["id"]
    assert any(r["id"] == row["id"] for r in db.latest_runs(limit=10))


def test_insert_and_read_back_a_prediction(cloud_credentials):
    from api import db

    row = db.insert_prediction(
        text_sha256="0" * 64,
        label="positive",
        probability_positive=0.9,
        confidence=0.9,
        calibrated=False,
        aspects=[{"aspect": "acting", "label": "positive", "score": 0.8, "evidence": []}],
        model_name="pytest",
        model_version="pytest",
        char_count=64,
    )
    assert row is not None, (
        "insert_prediction returned None — the predictions table is missing or "
        "the service key cannot write to it."
    )
    recent = db.latest_predictions(limit=10)
    assert any(r["id"] == row["id"] for r in recent)

    # The aspect breakdown must survive the round trip as structured JSON, not
    # as a string. If this fails, the column is `text` instead of `jsonb`.
    stored = next(r for r in recent if r["id"] == row["id"])
    assert isinstance(stored["aspects"], list)
    assert stored["aspects"][0]["aspect"] == "acting"


def test_the_label_check_constraint_is_real(cloud_credentials):
    """The database refuses a label the schema does not allow.

    insert_prediction swallows the error and returns None (logging must never
    take a request down), so a None here is the constraint doing its job.
    """
    from api import db

    assert (
        db.insert_prediction(
            text_sha256="1" * 64,
            label="ecstatic",
            probability_positive=0.9,
            confidence=0.9,
            calibrated=False,
            aspects=[],
            model_name="pytest",
            model_version="pytest",
        )
        is None
    )
