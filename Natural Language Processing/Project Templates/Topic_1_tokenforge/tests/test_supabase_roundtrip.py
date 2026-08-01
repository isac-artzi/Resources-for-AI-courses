"""The one test that touches the real cloud.

It is SKIPPED unless SUPABASE_URL and SUPABASE_SERVICE_KEY are set, so the suite
stays green offline. Run it once after you apply db/migrations/001_init.sql, to
prove the third cloud is actually wired up:

    SUPABASE_URL=... SUPABASE_SERVICE_KEY=... pytest -m cloud
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.cloud


def test_insert_and_read_back_a_run(cloud_credentials):
    from api import db

    row = db.insert_run(
        kind="preprocess",
        text_sha256="0" * 64,
        config={"test": True},
        model_version="pytest",
        token_count_before=10,
        token_count_after=6,
    )
    assert row is not None, (
        "insert_run returned None — check SUPABASE_URL/SUPABASE_SERVICE_KEY and "
        "that you applied db/migrations/001_init.sql."
    )
    assert row["id"]

    recent = db.latest_runs(limit=10)
    assert any(r["id"] == row["id"] for r in recent)
