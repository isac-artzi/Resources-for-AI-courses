"""Shared pytest fixtures.

FINISHED CODE. The whole suite runs offline: the Supabase layer is replaced by
an in-memory fake, so you can develop and test the NLP layer with no cloud
account and no network. The one test that needs real credentials skips itself
when they are absent.

The fake is a little larger than Topic 1's because this product has a write
path. It keeps four stores — runs, extractions, entities, reviews — and it
enforces the rule the real schema enforces: a review is an INSERT into reviews
and never an UPDATE of an entity. If you change the fake to "simplify" that, the
tests will still pass and your audit trail will still be broken, which is the
kind of green suite that helps nobody.

Markers (declared in pytest.ini):
    contract  — checks a function in api/nlp.py that you must implement.
                These FAIL until you write the code. That is the point: they
                are your to-do list. Run `pytest -m contract` to see what is left.
    network   — downloads a model or a dataset.
    cloud     — needs real Supabase credentials; skipped otherwise.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture
def fake_db(monkeypatch):
    """Replace api.db with an in-memory store. Returns the stores for assertions."""
    from api import db

    stores: dict[str, list[dict]] = {
        "runs": [],
        "extractions": [],
        "entities": [],
        "reviews": [],
    }

    def insert_run(
        model_type,
        dataset,
        config,
        model_version,
        precision=None,
        recall=None,
        f1=None,
        metrics=None,
        notes=None,
    ):
        row = {
            "id": len(stores["runs"]) + 1,
            "model_type": model_type,
            "dataset": dataset,
            "config": config,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "metrics": metrics,
            "model_version": model_version,
            "notes": notes,
            "created_at": "2026-01-01T00:00:00Z",
        }
        stores["runs"].append(row)
        return row

    def insert_extraction(text_sha256, model, model_version, entity_count, latency_ms=None):
        row = {
            "id": len(stores["extractions"]) + 1,
            "text_sha256": text_sha256,
            "model": model,
            "model_version": model_version,
            "entity_count": entity_count,
            "latency_ms": latency_ms,
            "created_at": "2026-01-01T00:00:00Z",
        }
        stores["extractions"].append(row)
        return row

    def insert_entities(extraction_id, entities):
        rows = []
        for e in entities:
            row = {
                "id": len(stores["entities"]) + 1,
                "extraction_id": extraction_id,
                "text": e["text"],
                "start_char": e["start_char"],
                "end_char": e["end_char"],
                "entity_type": e["entity_type"],
                "confidence": e["confidence"],
                "context": e.get("context"),
                "created_at": "2026-01-01T00:00:00Z",
            }
            stores["entities"].append(row)
            rows.append(row)
        return rows

    def get_entity(entity_id):
        for row in stores["entities"]:
            if row["id"] == entity_id:
                parent = next(
                    (x for x in stores["extractions"] if x["id"] == row["extraction_id"]), {}
                )
                return {
                    **row,
                    "extractions": {
                        "model": parent.get("model"),
                        "model_version": parent.get("model_version"),
                        "text_sha256": parent.get("text_sha256"),
                    },
                }
        return None

    def review_queue(threshold, limit=50, include_reviewed=False):
        reviewed = {r["entity_id"] for r in stores["reviews"]}
        rows = [e for e in stores["entities"] if e["confidence"] < threshold]
        if not include_reviewed:
            rows = [e for e in rows if e["id"] not in reviewed]
        rows.sort(key=lambda e: e["confidence"])
        out = []
        for row in rows[:limit]:
            parent = next(
                (x for x in stores["extractions"] if x["id"] == row["extraction_id"]), {}
            )
            out.append(
                {
                    **row,
                    "extractions": {
                        "model": parent.get("model"),
                        "model_version": parent.get("model_version"),
                    },
                }
            )
        return out

    def insert_review(
        entity_id,
        reviewer_id,
        decision,
        original_type,
        original_start_char,
        original_end_char,
        original_confidence,
        corrected_type=None,
        corrected_start_char=None,
        corrected_end_char=None,
        note=None,
    ):
        row = {
            "id": len(stores["reviews"]) + 1,
            "entity_id": entity_id,
            "reviewer_id": reviewer_id,
            "decision": decision,
            "original_type": original_type,
            "original_start_char": original_start_char,
            "original_end_char": original_end_char,
            "original_confidence": original_confidence,
            "corrected_type": corrected_type,
            "corrected_start_char": corrected_start_char,
            "corrected_end_char": corrected_end_char,
            "note": note,
            "created_at": "2026-01-01T00:00:00Z",
        }
        stores["reviews"].append(row)
        # Deliberately does NOT touch stores["entities"]. The prediction stays
        # exactly as the model made it. See api/db.insert_review.
        return row

    monkeypatch.setattr(db, "configured", lambda: True)
    monkeypatch.setattr(db, "ping", lambda: True)
    monkeypatch.setattr(db, "insert_run", insert_run)
    monkeypatch.setattr(
        db, "latest_runs", lambda limit=50: list(reversed(stores["runs"]))[:limit]
    )
    monkeypatch.setattr(db, "insert_extraction", insert_extraction)
    monkeypatch.setattr(db, "insert_entities", insert_entities)
    monkeypatch.setattr(db, "get_entity", get_entity)
    monkeypatch.setattr(db, "review_queue", review_queue)
    monkeypatch.setattr(db, "insert_review", insert_review)
    monkeypatch.setattr(
        db, "latest_reviews", lambda limit=50: list(reversed(stores["reviews"]))[:limit]
    )
    return stores


@pytest.fixture
def client(fake_db):
    """A FastAPI TestClient with the database faked out."""
    from fastapi.testclient import TestClient

    from api.main import app

    return TestClient(app)


@pytest.fixture
def seeded_entity(fake_db):
    """One logged low-confidence entity, so the review path has something to act on.

    Returns the entity row. This is what a real extraction would have written if
    your model had run — which is why the review tests do not need your NLP code
    to work yet.
    """
    from api import db

    extraction = db.insert_extraction(
        text_sha256="0" * 64,
        model="transformer",
        model_version="pytest",
        entity_count=1,
        latency_ms=12,
    )
    rows = db.insert_entities(
        extraction["id"],
        [
            {
                "text": "Ada Lovelace",
                "start_char": 4,
                "end_char": 16,
                "entity_type": "ORG",  # wrong on purpose; a reviewer will fix it
                "confidence": 0.41,
                "context": None,
            }
        ],
    )
    return rows[0]


@pytest.fixture(scope="session")
def cloud_credentials():
    """Real Supabase credentials, or a skip."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not (url and key):
        pytest.skip("SUPABASE_URL / SUPABASE_SERVICE_KEY not set — skipping cloud test.")
    return url, key
