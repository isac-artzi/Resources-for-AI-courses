"""Shared pytest fixtures.

FINISHED CODE. The whole suite runs offline: the Supabase layer is replaced by
an in-memory fake, so you can develop and test the generation layer with no cloud
account. The one test that needs real credentials skips itself when they are
absent, and the tests that need a downloaded decoder are marked so you can
deselect them.

Markers (declared in pytest.ini):
    contract  — checks a function in api/nlp.py that you must implement.
                These FAIL until you write the code. That is the point: they
                are your to-do list. Run `pytest -m contract` to see what is left.
    network   — downloads a model or tokenizer from Hugging Face.
    train     — actually fine-tunes; slow. Deselect it while you iterate.
    cloud     — needs real Supabase credentials; skipped otherwise.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture
def fake_db(monkeypatch):
    """Replace api.db with an in-memory store.

    Returns a dict with two lists, ``generations`` and ``training_runs``, so a
    test can assert on what the route wrote — including that the raw prompt is
    not in there.
    """
    from api import db

    store: dict[str, list[dict]] = {"generations": [], "training_runs": []}

    def insert_generation(
        prompt_sha256,
        strategy,
        decoding_params,
        generated_text,
        model_version,
        prompt_token_count=None,
        generated_token_count=None,
        distinct_1=None,
        distinct_2=None,
        perplexity=None,
        latency_ms=None,
    ):
        row = {
            "id": len(store["generations"]) + 1,
            "prompt_sha256": prompt_sha256,
            "strategy": strategy,
            "decoding_params": decoding_params,
            "generated_text": generated_text,
            "model_version": model_version,
            "prompt_token_count": prompt_token_count,
            "generated_token_count": generated_token_count,
            "distinct_1": distinct_1,
            "distinct_2": distinct_2,
            "perplexity": perplexity,
            "latency_ms": latency_ms,
            "human_rating": None,
            "ratings": [],
            "created_at": "2026-01-01T00:00:00Z",
        }
        store["generations"].append(row)
        return row

    def insert_training_run(
        base_model,
        model_version,
        hyperparameters,
        corpus_source=None,
        corpus_sha256=None,
        corpus_sentence_count=None,
        held_out_perplexity=None,
        notes=None,
    ):
        row = {
            "id": len(store["training_runs"]) + 1,
            "base_model": base_model,
            "model_version": model_version,
            "hyperparameters": hyperparameters,
            "corpus_source": corpus_source,
            "corpus_sha256": corpus_sha256,
            "corpus_sentence_count": corpus_sentence_count,
            "held_out_perplexity": held_out_perplexity,
            "notes": notes,
            "created_at": "2026-01-01T00:00:00Z",
        }
        store["training_runs"].append(row)
        return row

    def get_generation(generation_id):
        for row in store["generations"]:
            if row["id"] == generation_id:
                return row
        return None

    def append_rating(generation_id, rating_payload):
        row = get_generation(generation_id)
        if row is None:
            return None
        row["ratings"].append(rating_payload)
        scores = [r["rating"] for r in row["ratings"] if isinstance(r.get("rating"), int)]
        row["human_rating"] = round(sum(scores) / len(scores)) if scores else None
        return row

    monkeypatch.setattr(db, "configured", lambda: True)
    monkeypatch.setattr(db, "ping", lambda: True)
    monkeypatch.setattr(db, "insert_generation", insert_generation)
    monkeypatch.setattr(db, "insert_training_run", insert_training_run)
    monkeypatch.setattr(db, "get_generation", get_generation)
    monkeypatch.setattr(db, "append_rating", append_rating)
    monkeypatch.setattr(
        db,
        "latest_generations",
        lambda limit=50: list(reversed(store["generations"]))[:limit],
    )
    monkeypatch.setattr(
        db,
        "latest_training_runs",
        lambda limit=20: list(reversed(store["training_runs"]))[:limit],
    )
    return store


@pytest.fixture
def client(fake_db):
    """A FastAPI TestClient with the database faked out."""
    from fastapi.testclient import TestClient

    from api.main import app

    return TestClient(app)


@pytest.fixture
def sample_corpus(tmp_path):
    """A tiny corpus file: one sentence per line, with a duplicate and a blank.

    Far below the 20,000-sentence floor the assignment sets — this is for testing
    that ``load_corpus`` reads, filters and counts, not for training anything.
    """
    path = tmp_path / "corpus.txt"
    path.write_text(
        "The harbour was quiet after the storm.\n"
        "Gulls picked over what the water had left on the slipway.\n"
        "The harbour was quiet after the storm.\n"
        "\n"
        "By noon the boats were out again.\n",
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture(scope="session")
def cloud_credentials():
    """Real Supabase credentials, or a skip."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not (url and key):
        pytest.skip("SUPABASE_URL / SUPABASE_SERVICE_KEY not set — skipping cloud test.")
    return url, key
