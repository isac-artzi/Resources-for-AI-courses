"""The one test that touches the real cloud.

It is SKIPPED unless SUPABASE_URL and SUPABASE_SERVICE_KEY are set, so the suite
stays green offline. Run it once after you apply db/migrations/001_init.sql, to
prove the third cloud is actually wired up:

    SUPABASE_URL=... SUPABASE_SERVICE_KEY=... pytest -m cloud

It exercises both tables and the rating write path, because those are the three
things the assignment says have to persist.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.cloud


def test_insert_and_read_back_a_generation(cloud_credentials):
    from api import db

    row = db.insert_generation(
        prompt_sha256="0" * 64,
        strategy="top_p",
        decoding_params={"strategy": "top_p", "top_p": 0.9, "max_new_tokens": 32},
        generated_text="pytest wrote this row",
        model_version="pytest",
        prompt_token_count=4,
        generated_token_count=5,
        distinct_2=1.0,
    )
    assert row is not None, (
        "insert_generation returned None — check SUPABASE_URL/SUPABASE_SERVICE_KEY "
        "and that you applied db/migrations/001_init.sql."
    )
    assert row["id"]

    recent = db.latest_generations(limit=10)
    assert any(r["id"] == row["id"] for r in recent)


def test_two_raters_both_survive(cloud_credentials):
    """The write path the human evaluation depends on.

    Two independent scores on one generation must produce two entries, not one
    overwritten by the other. If this fails, your disagreement analysis has no
    data behind it.
    """
    from api import db

    row = db.insert_generation(
        prompt_sha256="1" * 64,
        strategy="greedy",
        decoding_params={"strategy": "greedy", "max_new_tokens": 16},
        generated_text="a row for the rating test",
        model_version="pytest",
    )
    assert row is not None

    now = datetime.now(timezone.utc).isoformat()
    db.append_rating(
        row["id"],
        {"rater_id": "rater-a", "rating": 2, "dimensions": {}, "notes": None,
         "recorded_at": now},
    )
    updated = db.append_rating(
        row["id"],
        {"rater_id": "rater-b", "rating": 4, "dimensions": {}, "notes": None,
         "recorded_at": now},
    )
    assert updated is not None
    assert len(updated["ratings"]) == 2
    assert updated["human_rating"] == 3  # rounded mean of 2 and 4


def test_insert_and_read_back_a_training_run(cloud_credentials):
    from api import db

    row = db.insert_training_run(
        base_model="gpt2",
        model_version="pytest-v0",
        hyperparameters={"epochs": 1, "learning_rate": 5e-5, "method": "full"},
        corpus_source="pytest",
        corpus_sha256="2" * 64,
        corpus_sentence_count=20000,
        held_out_perplexity=42.0,
    )
    assert row is not None, (
        "insert_training_run returned None — did you apply the WHOLE migration? "
        "The training_runs table is the second half of it."
    )
    assert any(r["id"] == row["id"] for r in db.latest_training_runs(limit=10))
