"""The one test that touches the real cloud.

It is SKIPPED unless SUPABASE_URL and SUPABASE_SERVICE_KEY are set, so the suite
stays green offline. Run it once after you apply db/migrations/001_init.sql, to
prove the third cloud is actually wired up:

    SUPABASE_URL=... SUPABASE_SERVICE_KEY=... pytest -m cloud

It walks the whole chain — extraction, entity, review — because the interesting
failure is not "can I insert a row", it is "does the foreign key hold and does
the review keep the prediction intact".
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.cloud


def test_insert_and_read_back_a_training_run(cloud_credentials):
    from api import db

    row = db.insert_run(
        model_type="crf",
        dataset="pytest",
        config={"test": True, "features": ["suffix3", "pos"]},
        model_version="pytest",
        precision=0.5,
        recall=0.5,
        f1=0.5,
    )
    assert row is not None, (
        "insert_run returned None — check SUPABASE_URL/SUPABASE_SERVICE_KEY and "
        "that you applied db/migrations/001_init.sql."
    )
    assert row["id"]

    recent = db.latest_runs(limit=10)
    assert any(r["id"] == row["id"] for r in recent)


def test_extraction_entity_review_chain_holds(cloud_credentials):
    from api import db

    extraction = db.insert_extraction(
        text_sha256="0" * 64,
        model="transformer",
        model_version="pytest",
        entity_count=1,
        latency_ms=5,
    )
    assert extraction is not None, "check that the extractions table exists"

    entities = db.insert_entities(
        extraction["id"],
        [
            {
                "text": "Ada Lovelace",
                "start_char": 0,
                "end_char": 12,
                "entity_type": "ORG",
                "confidence": 0.42,
            }
        ],
    )
    assert entities, "check that the entities table exists and the FK is satisfied"
    entity_id = entities[0]["id"]

    queue = db.review_queue(threshold=0.85, limit=100)
    assert any(r["id"] == entity_id for r in queue), (
        "a 0.42-confidence entity should be below a 0.85 threshold — if it is not "
        "in the queue, check the confidence column type and the RLS policies"
    )

    review = db.insert_review(
        entity_id=entity_id,
        reviewer_id="pytest",
        decision="correct",
        original_type="ORG",
        original_start_char=0,
        original_end_char=12,
        original_confidence=0.42,
        corrected_type="PER",
        note="round-trip test",
    )
    assert review is not None
    assert review["original_type"] == "ORG"

    # The prediction is unchanged; the correction lives in its own row.
    stored = db.get_entity(entity_id)
    assert stored["entity_type"] == "ORG"

    # And the reviewed entity has left the queue.
    queue_after = db.review_queue(threshold=0.85, limit=100)
    assert not any(r["id"] == entity_id for r in queue_after)
