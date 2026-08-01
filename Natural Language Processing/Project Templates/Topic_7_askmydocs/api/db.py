"""Supabase persistence for the API tier — Cloud #3.

FINISHED CODE. You should not need to change anything here to complete the
assignment, but you should read it, because it shows the three rules the whole
three-cloud split depends on:

1. Only the API talks to Supabase with the SERVICE-ROLE key. The Streamlit tier
   never imports this module; it does its own read-only query with the ANON key.
2. Nothing here writes a raw question. Callers pass a sha256 hash. The passages
   ARE stored in full, because a retrieval-augmented answer cannot be audited
   without them — which is exactly why a public demo should not be pointed at a
   private document collection.
3. Logging failures never fail the request. If Supabase is unreachable the user
   still gets their answer, just without a query_id and without an audit row.

The five tables mirror the pipeline: ``documents`` -> ``chunks`` (one vector
each) and then, per question, ``queries`` -> ``retrievals`` -> ``answers``. The
last three are the audit trail the product brief asks for: what was asked, what
came back for it, and what the model then said.

Environment variables (a .env locally, the Render dashboard in production):
    SUPABASE_URL          -> https://<project-ref>.supabase.co
    SUPABASE_SERVICE_KEY  -> service-role key. Secret. Never in the UI tier.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from supabase import Client, create_client

_client: Optional[Client] = None


def configured() -> bool:
    """True when both Supabase environment variables are present.

    The API stays up without them so you can develop the NLP layer offline; it
    just skips logging and returns query_id=None. /healthz reports the difference
    so you can tell "misconfigured" from "broken".
    """
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"))


def get_client() -> Client:
    """Lazily create and cache the Supabase client."""
    global _client
    if _client is None:
        _client = create_client(
            os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"]
        )
    return _client


def ping() -> bool:
    """True if the documents table is reachable. Used by /healthz."""
    if not configured():
        return False
    try:
        get_client().table("documents").select("id").limit(1).execute()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Ingest side: documents and chunks
# ---------------------------------------------------------------------------
def document_hashes(corpus: str) -> set[str]:
    """Every content hash already stored in one corpus.

    This is the cheap half of the disjointness guarantee. Before ingesting into
    the retrieval collection, api/main.py asks for the fine-tuning corpus's
    hashes and refuses any document that is already in there — and the reverse.
    The database has a unique index on content_sha256 that refuses it a second
    time, so a bug in the API cannot contaminate the experiment.
    """
    if not configured():
        return set()
    try:
        resp = (
            get_client()
            .table("documents")
            .select("content_sha256")
            .eq("corpus", corpus)
            .execute()
        )
        return {r["content_sha256"] for r in (resp.data or [])}
    except Exception:
        return set()


def insert_document(
    title: str,
    source: str,
    corpus: str,
    content_sha256: str,
    token_count: Optional[int] = None,
    doc_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[dict]:
    """Register one document. Returns the inserted row, or None.

    A unique index on content_sha256 means re-ingesting the same document raises
    inside PostgREST rather than creating a duplicate. That is intentional: two
    copies of the same document in the collection both get retrieved, both fill
    a slot in your top-5, and your k=5 quietly becomes a k=4.
    """
    if not configured():
        return None
    try:
        resp = (
            get_client()
            .table("documents")
            .insert(
                {
                    "title": title,
                    "source": source,
                    "corpus": corpus,
                    "content_sha256": content_sha256,
                    "token_count": token_count,
                    "doc_metadata": doc_metadata or {},
                }
            )
            .execute()
        )
        return resp.data[0] if resp.data else None
    except Exception:
        return None


def insert_chunks(rows: Sequence[Dict[str, Any]]) -> List[dict]:
    """Bulk-insert chunk rows (text, offsets, chunking parameters, embedding).

    Each row should carry: document_id, ordinal, content, token_count,
    start_token, end_token, chunk_size_tokens, overlap_tokens, tokenizer_name,
    embedding (a list of floats), embedding_model, embedding_dim.

    The chunking parameters are stored PER CHUNK rather than once per ingest on
    purpose. You will re-chunk. When half the collection is at 400/15 percent and
    half is at 350/10 percent because you changed your mind on a Tuesday, the
    only way to explain a retrieval result is a row that says which settings
    produced it.

    Returns the inserted rows, or [] on failure. Inserts in batches because a
    single request with 5,000 384-dimensional vectors in it will time out.
    """
    if not configured() or not rows:
        return []
    inserted: List[dict] = []
    batch_size = 200
    try:
        client = get_client()
        for start in range(0, len(rows), batch_size):
            batch = list(rows[start : start + batch_size])
            resp = client.table("chunks").insert(batch).execute()
            inserted.extend(resp.data or [])
        return inserted
    except Exception:
        return inserted


def list_sources(limit: int = 200) -> List[dict]:
    """Documents with their chunk counts — the data source for GET /sources."""
    if not configured():
        return []
    try:
        resp = (
            get_client()
            .table("documents")
            .select("id, title, source, corpus, token_count, chunks(count)")
            .order("id")
            .limit(limit)
            .execute()
        )
    except Exception:
        return []

    out: List[dict] = []
    for row in resp.data or []:
        nested = row.get("chunks") or []
        # PostgREST returns an aggregate as [{"count": n}]; be tolerant of both
        # that shape and a plain list of rows.
        if nested and isinstance(nested[0], dict) and "count" in nested[0]:
            chunk_count = nested[0]["count"]
        else:
            chunk_count = len(nested)
        out.append(
            {
                "document_id": row["id"],
                "title": row["title"],
                "source": row["source"],
                "corpus": row["corpus"],
                "token_count": row.get("token_count"),
                "chunk_count": chunk_count,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Query side: the vector search itself
# ---------------------------------------------------------------------------
def match_chunks(
    embedding: Sequence[float],
    k: int = 5,
    corpus: str = "retrieval",
) -> List[dict]:
    """Top-k nearest chunks by cosine distance, computed inside Postgres.

    Calls the ``match_chunks`` SQL function created by
    db/migrations/001_init.sql, which uses pgvector's ``<=>`` operator and the
    ivfflat index. Returns rows with: chunk_id, document_id, document_title,
    content, similarity (already converted from distance: similarity =
    1 - distance).

    Doing the search in the database rather than pulling every vector into Python
    is not an optimisation detail — with 5,000 chunks the pull-everything version
    works fine in development and falls over the first time a grader opens the
    app on a free-plan instance with 512 MB of memory.

    Returns [] when Supabase is unreachable, so a dead database degrades to
    "no passages found" rather than a 500.
    """
    if not configured():
        return []
    try:
        resp = (
            get_client()
            .rpc(
                "match_chunks",
                {
                    "query_embedding": list(embedding),
                    "match_count": k,
                    "corpus_filter": corpus,
                },
            )
            .execute()
        )
        return resp.data or []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Audit side: queries, retrievals, answers
# ---------------------------------------------------------------------------
def insert_query(
    query_sha256: str,
    k: int,
    model_version: str,
    embedding_model: Optional[str] = None,
) -> Optional[dict]:
    """Log that a question was asked. The question itself is never stored.

    A knowledge-management team's questions leak more than their documents do.
    "Which suppliers are under review?" is a fact about the company whether or
    not it is ever answered. The hash gives you de-duplication and reproducibility
    without keeping the text.
    """
    if not configured():
        return None
    try:
        resp = (
            get_client()
            .table("queries")
            .insert(
                {
                    "query_sha256": query_sha256,
                    "k": k,
                    "model_version": model_version,
                    "embedding_model": embedding_model,
                }
            )
            .execute()
        )
        return resp.data[0] if resp.data else None
    except Exception:
        return None


def insert_retrievals(query_id: int, retrieved: Sequence[Any]) -> List[dict]:
    """Log which chunks came back for a query, with similarity and rank.

    ``retrieved`` is a sequence of RetrievedChunk models (or anything with
    chunk_id / similarity / rank attributes).

    This table is the one that makes the Retrieval Audit tab possible, and it is
    the one that separates "the retriever never found it" from "the retriever
    found it and the generator ignored it". Those two produce the same wrong
    answer and have completely different fixes: the first is a chunking or
    embedding problem, the second is a prompting problem. Without this table you
    cannot tell which afternoon to spend.
    """
    if not configured() or not retrieved:
        return []
    try:
        rows = [
            {
                "query_id": query_id,
                "chunk_id": getattr(r, "chunk_id", None),
                "similarity": getattr(r, "similarity", None),
                "rank": getattr(r, "rank", None),
            }
            for r in retrieved
        ]
        resp = get_client().table("retrievals").insert(rows).execute()
        return resp.data or []
    except Exception:
        return []


def insert_answer(
    query_id: int,
    answer: str,
    retrieval_used: bool,
    cited_chunk_ids: Sequence[int],
    generator_model: str,
) -> Optional[dict]:
    """Log the generated answer, whether retrieval was used, and what it cited.

    ``retrieval_used`` is what turns the log into an experiment: filter this
    table on it and you have both halves of the with/without comparison, keyed by
    the same query hash.
    """
    if not configured():
        return None
    try:
        resp = (
            get_client()
            .table("answers")
            .insert(
                {
                    "query_id": query_id,
                    "answer": answer,
                    "retrieval_used": retrieval_used,
                    "cited_chunk_ids": list(cited_chunk_ids),
                    "generator_model": generator_model,
                }
            )
            .execute()
        )
        return resp.data[0] if resp.data else None
    except Exception:
        return None


def latest_retrievals(limit: int = 100) -> List[dict]:
    """Recent retrieval rows joined to their chunk and query.

    Backs the Retrieval Audit tab when it falls back to the API instead of
    reading Postgres directly. Newest first.
    """
    if not configured():
        return []
    try:
        resp = (
            get_client()
            .table("retrievals")
            .select(
                "id, query_id, chunk_id, similarity, rank, created_at, "
                "chunks(document_id, ordinal, content), "
                "queries(query_sha256, k, model_version, created_at)"
            )
            .order("id", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception:
        return []
