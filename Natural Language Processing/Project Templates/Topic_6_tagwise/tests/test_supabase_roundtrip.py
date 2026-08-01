"""The one test that touches the real cloud.

It is SKIPPED unless SUPABASE_URL and SUPABASE_SERVICE_KEY are set, so the suite
stays green offline. Run it once after you apply db/migrations/001_init.sql, to
prove the third cloud is actually wired up:

    SUPABASE_URL=... SUPABASE_SERVICE_KEY=... pytest -m cloud

It writes to BOTH tables, because applying half the migration is a real failure
mode: `runs` works, the comparison tab fills in, and the History tab stays empty
for a week while you look for the bug in your UI code.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.cloud


def test_insert_and_read_back_a_run(cloud_credentials):
    from api import db

    row = db.insert_run(
        model="baseline",
        tagset="UPOS",
        hyperparameters={"test": True},
        model_version="pytest",
        accuracy=0.9,
        macro_f1=0.7,
        metrics={"confusion": {"labels": ["NOUN", "VERB"], "matrix": [[1, 0], [0, 1]]}},
    )
    assert row is not None, (
        "insert_run returned None — check SUPABASE_URL/SUPABASE_SERVICE_KEY and "
        "that you applied db/migrations/001_init.sql."
    )
    assert row["id"]

    recent = db.latest_runs(limit=10)
    assert any(r["id"] == row["id"] for r in recent)


def test_insert_and_read_back_a_tagging(cloud_credentials):
    from api import db

    row = db.insert_tagging(
        sentence_sha256="0" * 64,
        token_count=3,
        tag_sequence=["PRON", "VERB", "PUNCT"],
        model="baseline",
        model_version="pytest",
        unknown_count=1,
    )
    assert row is not None, (
        "insert_tagging returned None — the taggings table is the one the History "
        "tab reads. Re-run the whole migration file if only runs exists."
    )
    assert row["tag_sequence"] == ["PRON", "VERB", "PUNCT"]

    recent = db.latest_taggings(limit=10)
    assert any(r["id"] == row["id"] for r in recent)
