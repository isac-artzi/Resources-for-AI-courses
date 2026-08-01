"""Shared pytest fixtures.

FINISHED CODE. The whole suite runs offline: the Supabase layer is replaced by
an in-memory fake, so you can develop and test the NLP layer with no cloud
account and no network. The one test that needs real credentials skips itself
when they are absent.

The fake store is a dict of five lists, one per table, and it enforces the two
rules that matter for the tests: content hashes are unique across corpora (the
disjointness guarantee), and inserted rows come back with ids. It does NOT do
vector search — that needs pgvector — so anything touching similarity is either
a pure-Python test or monkeypatched.

Markers (declared in pytest.ini):
    contract  — checks a function in api/nlp.py that you must implement.
                These FAIL until you write the code. That is the point: they
                are your to-do list. Run `pytest -m contract` to see what is left.
    network   — downloads a model from Hugging Face; deselect with
                -m "not network".
    cloud     — needs real Supabase credentials; skipped otherwise.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture
def fake_db(monkeypatch):
    """Replace api.db with an in-memory store. Returns the store for assertions."""
    from api import db

    store: dict[str, list] = {
        "documents": [],
        "chunks": [],
        "queries": [],
        "retrievals": [],
        "answers": [],
    }

    def document_hashes(corpus):
        return {d["content_sha256"] for d in store["documents"] if d["corpus"] == corpus}

    def insert_document(
        title, source, corpus, content_sha256, token_count=None, doc_metadata=None
    ):
        # The unique index on content_sha256 lives in the migration; the fake
        # honours it so a test cannot pass here and fail against real Postgres.
        if any(d["content_sha256"] == content_sha256 for d in store["documents"]):
            return None
        row = {
            "id": len(store["documents"]) + 1,
            "title": title,
            "source": source,
            "corpus": corpus,
            "content_sha256": content_sha256,
            "token_count": token_count,
            "doc_metadata": doc_metadata or {},
            "created_at": "2026-01-01T00:00:00Z",
        }
        store["documents"].append(row)
        return row

    def insert_chunks(rows):
        out = []
        for r in rows:
            row = dict(r)
            row["id"] = len(store["chunks"]) + 1
            store["chunks"].append(row)
            out.append(row)
        return out

    def list_sources(limit=200):
        return [
            {
                "document_id": d["id"],
                "title": d["title"],
                "source": d["source"],
                "corpus": d["corpus"],
                "token_count": d["token_count"],
                "chunk_count": sum(
                    1 for c in store["chunks"] if c["document_id"] == d["id"]
                ),
            }
            for d in store["documents"][:limit]
        ]

    def match_chunks(embedding, k=5, corpus="retrieval"):
        return []

    def insert_query(query_sha256, k, model_version, embedding_model=None):
        row = {
            "id": len(store["queries"]) + 1,
            "query_sha256": query_sha256,
            "k": k,
            "model_version": model_version,
            "embedding_model": embedding_model,
            "created_at": "2026-01-01T00:00:00Z",
        }
        store["queries"].append(row)
        return row

    def insert_retrievals(query_id, retrieved):
        out = []
        for r in retrieved:
            row = {
                "id": len(store["retrievals"]) + 1,
                "query_id": query_id,
                "chunk_id": getattr(r, "chunk_id", None),
                "similarity": getattr(r, "similarity", None),
                "rank": getattr(r, "rank", None),
            }
            store["retrievals"].append(row)
            out.append(row)
        return out

    def insert_answer(query_id, answer, retrieval_used, cited_chunk_ids, generator_model):
        row = {
            "id": len(store["answers"]) + 1,
            "query_id": query_id,
            "answer": answer,
            "retrieval_used": retrieval_used,
            "cited_chunk_ids": list(cited_chunk_ids),
            "generator_model": generator_model,
        }
        store["answers"].append(row)
        return row

    monkeypatch.setattr(db, "configured", lambda: True)
    monkeypatch.setattr(db, "ping", lambda: True)
    monkeypatch.setattr(db, "document_hashes", document_hashes)
    monkeypatch.setattr(db, "insert_document", insert_document)
    monkeypatch.setattr(db, "insert_chunks", insert_chunks)
    monkeypatch.setattr(db, "list_sources", list_sources)
    monkeypatch.setattr(db, "match_chunks", match_chunks)
    monkeypatch.setattr(db, "insert_query", insert_query)
    monkeypatch.setattr(db, "insert_retrievals", insert_retrievals)
    monkeypatch.setattr(db, "insert_answer", insert_answer)
    monkeypatch.setattr(db, "latest_retrievals", lambda limit=100: list(
        reversed(store["retrievals"])
    )[:limit])
    return store


@pytest.fixture
def client(fake_db):
    """A FastAPI TestClient with the database faked out."""
    from fastapi.testclient import TestClient

    from api.main import app

    return TestClient(app)


@pytest.fixture(scope="session")
def cloud_credentials():
    """Real Supabase credentials, or a skip."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not (url and key):
        pytest.skip("SUPABASE_URL / SUPABASE_SERVICE_KEY not set — skipping cloud test.")
    return url, key
