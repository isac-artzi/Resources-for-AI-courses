"""The one test that touches the real cloud.

It is SKIPPED unless SUPABASE_URL and SUPABASE_SERVICE_KEY are set, so the suite
stays green offline. Run it once after you apply db/migrations/001_init.sql, to
prove the third cloud is actually wired up:

    SUPABASE_URL=... SUPABASE_SERVICE_KEY=... pytest -m cloud

It writes a document, a chunk, a query, a retrieval and an answer, then reads
them back and deletes the document — the foreign keys cascade, so removing the
document removes its chunks, and removing the query removes its retrievals and
answer.

If the first assertion fails, the usual cause is that the migration stopped at
`create extension if not exists vector;`. Check Database → Extensions in the
Supabase dashboard: pgvector ships with the project but is not on by default, and
without it the chunks table was never created.
"""
from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.cloud


def test_ingest_and_audit_round_trip(cloud_credentials):
    from api import db
    from shared.schemas import RetrievedChunk

    digest = uuid.uuid4().hex * 2  # 64 hex chars, guaranteed not to collide
    doc = db.insert_document(
        title="pytest document",
        source="tests/test_supabase_roundtrip.py",
        corpus="retrieval",
        content_sha256=digest,
        token_count=12,
        doc_metadata={"test": True},
    )
    assert doc is not None, (
        "insert_document returned None — check SUPABASE_URL/SUPABASE_SERVICE_KEY "
        "and that you applied db/migrations/001_init.sql."
    )

    chunks = db.insert_chunks(
        [
            {
                "document_id": doc["id"],
                "ordinal": 0,
                "content": "A passage written by the test suite.",
                "token_count": 8,
                "start_token": 0,
                "end_token": 8,
                "chunk_size_tokens": 400,
                "overlap_tokens": 60,
                "tokenizer_name": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding": None,
                "embedding_model": None,
                "embedding_dim": None,
            }
        ]
    )
    assert chunks, "insert_chunks returned nothing — is the chunks table present?"

    query = db.insert_query(
        query_sha256="0" * 64, k=5, model_version="pytest", embedding_model="pytest"
    )
    assert query is not None

    rows = db.insert_retrievals(
        query["id"],
        [
            RetrievedChunk(
                chunk_id=chunks[0]["id"],
                document_id=doc["id"],
                document_title="pytest document",
                text="A passage written by the test suite.",
                similarity=0.77,
                rank=1,
            )
        ],
    )
    assert rows, "insert_retrievals returned nothing — the audit trail is broken"

    answer = db.insert_answer(
        query_id=query["id"],
        answer="A test answer citing [1].",
        retrieval_used=True,
        cited_chunk_ids=[chunks[0]["id"]],
        generator_model="pytest",
    )
    assert answer is not None

    sources = db.list_sources(limit=200)
    assert any(s["document_id"] == doc["id"] for s in sources)

    # Clean up. Cascades remove the chunk, the retrieval and the answer.
    client = db.get_client()
    client.table("documents").delete().eq("id", doc["id"]).execute()
    client.table("queries").delete().eq("id", query["id"]).execute()


def test_the_unique_index_refuses_the_same_document_in_both_corpora(cloud_credentials):
    """The database's half of the corpus-separation guarantee.

    api/main.py already refuses this with a 409. This test proves the database
    would refuse it too, which is what makes the guarantee survive a bug in the
    API — and it is the evidence to cite in your report when you are asked how
    you kept the two sets separate.
    """
    from api import db

    digest = uuid.uuid4().hex * 2
    first = db.insert_document(
        title="in the fine-tuning corpus",
        source="tests",
        corpus="finetune",
        content_sha256=digest,
    )
    assert first is not None

    second = db.insert_document(
        title="the same text, in the retrieval collection",
        source="tests",
        corpus="retrieval",
        content_sha256=digest,
    )
    assert second is None, (
        "The unique index on documents.content_sha256 is missing. Re-apply "
        "db/migrations/001_init.sql — without it, the same document can sit in "
        "both corpora and the with/without-retrieval comparison is meaningless."
    )

    db.get_client().table("documents").delete().eq("id", first["id"]).execute()
