"""FastAPI service — Cloud #2, deployed on Render.com.

FINISHED CODE, with one exception: the route bodies call into ``api.nlp``, which
is where your work lives. Read this file to see the shape of the service, then go
implement ``api/nlp.py``.

Design rules this file follows, which your report should be able to explain:

* No NLP logic here. Routes validate, delegate, log, and return. If you find
  yourself writing a similarity calculation in this file, it belongs in nlp.py.
* No UI logic here either. The service returns data; the Streamlit tier decides
  what it looks like.
* Logging failures never fail the request (see api/db.py).
* The corpus separation is enforced at the door, not in a README. ``/embed``
  refuses to ingest a document that already exists in the other corpus, and says
  which one, before a single vector is written.

Run locally:  uvicorn api.main:app --reload
Then open:    http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import os
import subprocess

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api import db, nlp
from shared.schemas import (
    AskRequest,
    AskResponse,
    EmbedRequest,
    EmbedResponse,
    Health,
    SourcesResponse,
    SourceSummary,
    Version,
)

SERVICE_NAME = "askmydocs-api"

# Bump this whenever you change the chunking parameters, the embedding model,
# the generator, or the prompt. Every logged query carries it, which is what lets
# you say "the answers improved at v3 because of the prompt, not because of the
# question" and have the rows to prove it.
MODEL_VERSION = os.environ.get("MODEL_VERSION", "askmydocs-v1")

app = FastAPI(
    title="AskMyDocs API",
    description=(
        "Retrieval-augmented question answering over your own document "
        "collection, on the three-cloud stack."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


def _git_sha() -> str:
    """Short SHA of the running build. Render sets RENDER_GIT_COMMIT for us."""
    if os.environ.get("RENDER_GIT_COMMIT"):
        return os.environ["RENDER_GIT_COMMIT"][:7]
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], timeout=2)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Infrastructure routes — required by the assignment, already working.
# ---------------------------------------------------------------------------
@app.get("/healthz", response_model=Health)
def healthz() -> Health:
    if not db.configured():
        return Health(status="ok", database="not_configured")
    return Health(status="ok", database="ok" if db.ping() else "unreachable")


@app.get("/version", response_model=Version)
def version() -> Version:
    return Version(service=SERVICE_NAME, git_sha=_git_sha(), model_version=MODEL_VERSION)


# ---------------------------------------------------------------------------
# Product routes — the bodies delegate to api/nlp.py (your work).
# ---------------------------------------------------------------------------
@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    """Chunk, embed, and store a batch of documents.

    The order of operations below is deliberate and you should not reorder it:
    every corpus-separation check runs BEFORE anything is written. A partial
    ingest that half-contaminated the retrieval collection is much worse than a
    refused one, because you cannot tell by looking which half is clean.
    """
    other_corpus = "finetune" if req.corpus == "retrieval" else "retrieval"

    # --- Guard 1: near-duplicate detection, when the caller supplies a sample of
    # the other corpus. Catches the reformatted copy that has a different hash.
    if req.finetune_sample:
        try:
            nlp.check_corpus_disjointness(
                req.finetune_sample, [d.text for d in req.documents]
            )
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except nlp.CorpusOverlapError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    # --- Guard 2: exact content hashes already stored in the other corpus.
    # Cheap, always on, and it is the check the database repeats with a unique
    # index on documents.content_sha256 in case this code is ever bypassed.
    existing = db.document_hashes(other_corpus)
    hashes = [nlp.sha256_text(d.text) for d in req.documents]
    for doc, digest in zip(req.documents, hashes):
        if digest in existing:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Refusing to ingest '{doc.title}' into the {req.corpus} corpus: "
                    f"the identical document is already in the {other_corpus} corpus. "
                    "The fine-tuning corpus and the retrieval collection must be "
                    "disjoint, or the with/without-retrieval comparison measures "
                    "nothing — a generator that memorised these documents during "
                    "fine-tuning will answer correctly without retrieval and "
                    "retrieval will appear to add zero. Remove the document from "
                    "one side and re-ingest."
                ),
            )

    # --- Guard 3: the stored vectors and the column must have the same width.
    # Checked here, at write time, because the alternative is finding out at
    # query time when a user asks a question and pgvector raises "different
    # vector dimensions" from inside a request.
    embedding_dim = nlp.EMBEDDING_DIM
    if req.corpus == "retrieval":
        try:
            embedding_dim = nlp.embedding_dimension(nlp.EMBEDDING_MODEL)
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        if embedding_dim != nlp.EMBEDDING_DIM:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"{nlp.EMBEDDING_MODEL} produces {embedding_dim}-dimensional "
                    f"vectors but this service is configured for {nlp.EMBEDDING_DIM} "
                    "and db/migrations/001_init.sql declares the column that width. "
                    "Change nlp.EMBEDDING_DIM, change the vector(...) column in the "
                    "migration, and re-embed the whole collection — you cannot mix "
                    "two widths in one column and old vectors cannot be converted."
                ),
            )

    all_chunk_ids: list[int] = []
    chunk_total = 0
    stored = db.configured()

    for doc, digest in zip(req.documents, hashes):
        try:
            chunks = nlp.chunk_document(doc, req.chunking)
            # The fine-tuning corpus is registered so the disjointness checks have
            # something to compare against, and so perplexity has a split to run
            # on. It is NOT embedded: nothing should ever be able to retrieve it.
            vectors = (
                nlp.embed_texts([c.text for c in chunks], nlp.EMBEDDING_MODEL)
                if req.corpus == "retrieval"
                else [None] * len(chunks)
            )
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        chunk_total += len(chunks)
        doc_row = db.insert_document(
            title=doc.title,
            source=doc.source,
            corpus=req.corpus,
            content_sha256=digest,
            token_count=sum(c.token_count for c in chunks),
            doc_metadata={"chunking": req.chunking.model_dump()},
        )
        if not doc_row:
            stored = False
            continue

        rows = [
            {
                "document_id": doc_row["id"],
                "ordinal": c.ordinal,
                "content": c.text,
                "token_count": c.token_count,
                "start_token": c.start_token,
                "end_token": c.end_token,
                "chunk_size_tokens": req.chunking.chunk_size_tokens,
                "overlap_tokens": req.chunking.overlap_tokens,
                "tokenizer_name": req.chunking.tokenizer_name,
                "embedding": vec,
                "embedding_model": nlp.EMBEDDING_MODEL if vec is not None else None,
                "embedding_dim": embedding_dim if vec is not None else None,
            }
            for c, vec in zip(chunks, vectors)
        ]
        all_chunk_ids.extend(r["id"] for r in db.insert_chunks(rows))

    return EmbedResponse(
        corpus=req.corpus,
        document_count=len(req.documents),
        chunk_count=chunk_total,
        embedding_model=nlp.EMBEDDING_MODEL,
        embedding_dim=embedding_dim,
        chunking=req.chunking,
        chunk_ids=all_chunk_ids,
        stored=stored,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Answer a question, with or without retrieval, and log all three tables."""
    try:
        result = nlp.answer_question(
            req.question, k=req.k, use_retrieval=req.use_retrieval
        )
    except NotImplementedError as exc:
        # A friendlier 501 than a 500 while you are still building.
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # The question itself is never stored — only its hash. See api/db.py.
    query_row = db.insert_query(
        query_sha256=nlp.sha256_text(req.question),
        k=req.k,
        model_version=MODEL_VERSION,
        embedding_model=result.embedding_model,
    )
    if query_row:
        result.query_id = query_row["id"]
        db.insert_retrievals(query_row["id"], result.retrieved)
        db.insert_answer(
            query_id=query_row["id"],
            answer=result.answer,
            retrieval_used=result.retrieval_used,
            cited_chunk_ids=result.cited_chunk_ids,
            generator_model=result.generator_model,
        )
    return result


@app.get("/sources", response_model=SourcesResponse)
def sources(limit: int = 200) -> SourcesResponse:
    """What is in the store — the citation targets, with their chunk counts.

    Both corpora are listed, with the ``corpus`` column saying which is which.
    Showing them together is on purpose: it is the fastest way for a grader (or
    for you) to confirm at a glance that no title appears on both sides.
    """
    rows = db.list_sources(limit=limit)
    items = [SourceSummary(**r) for r in rows]
    return SourcesResponse(
        sources=items,
        document_count=len(items),
        chunk_count=sum(i.chunk_count for i in items),
    )
