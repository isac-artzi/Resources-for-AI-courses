"""Infrastructure tests. These pass on a fresh fork — if they don't, fix the
environment before you write any NLP code."""
from __future__ import annotations

import hashlib


def test_healthz_reports_ok(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["database"] in {"ok", "unreachable", "not_configured"}


def test_version_exposes_build_identity(client):
    resp = client.get("/version")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "askmydocs-api"
    # Graders check this against the live URL in your README.
    assert body["git_sha"]
    assert body["model_version"]


def test_sources_endpoint_is_reachable_and_empty_at_first(client):
    resp = client.get("/sources")
    assert resp.status_code == 200
    body = resp.json()
    assert body["sources"] == []
    assert body["document_count"] == 0
    assert body["chunk_count"] == 0


def test_embed_refuses_a_document_already_in_the_finetuning_corpus(client, fake_db):
    """The corpus-separation guard, tested as infrastructure rather than trusted.

    This one passes on a fresh fork because the check runs before any function in
    api/nlp.py is called. That ordering is the point: the ingest is refused
    before a single vector is written, so there is no half-contaminated state to
    clean up afterwards.
    """
    text = "The escalation policy was revised in the third quarter."
    fake_db["documents"].append(
        {
            "id": 1,
            "title": "already fine-tuned on this",
            "source": "test",
            "corpus": "finetune",
            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "token_count": 9,
            "doc_metadata": {},
            "created_at": "2026-01-01T00:00:00Z",
        }
    )

    resp = client.post(
        "/embed",
        json={
            "documents": [{"title": "same doc", "source": "test", "text": text}],
            "corpus": "retrieval",
        },
    )
    assert resp.status_code == 409, (
        "A document in the fine-tuning corpus must not be ingestible into the "
        "retrieval collection. If this returns 501 the guard is running after "
        "the NLP stubs instead of before them."
    )
    detail = resp.json()["detail"]
    assert "disjoint" in detail.lower()
    assert fake_db["chunks"] == []


def test_ask_rejects_an_out_of_range_k_before_your_code_runs(client):
    resp = client.post("/ask", json={"question": "does k validate?", "k": 500})
    assert resp.status_code == 422
