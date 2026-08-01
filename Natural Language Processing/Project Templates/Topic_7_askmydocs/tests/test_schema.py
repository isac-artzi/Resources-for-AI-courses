"""Schema tests — the JSON contract between the two clouds.

These also pass on a fresh fork. They exist so that if you change
shared/schemas.py in a way that breaks the UI, you find out here rather than in
a deployed app.

Several of them are really tests of the assignment's own constraints: the 300-500
token chunk size, the 10-20 percent overlap, and the [-1, 1] similarity range are
enforced by pydantic, so a bad config is rejected at the edge of the service
instead of producing quietly mediocre retrieval.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.schemas import (
    AskRequest,
    ChunkingParams,
    EmbedRequest,
    RetrievedChunk,
)


def test_ask_defaults_are_the_documented_ones():
    req = AskRequest(question="what is the escalation policy?")
    assert req.k == 5
    assert req.use_retrieval is True


def test_empty_question_is_rejected_before_it_reaches_your_code():
    with pytest.raises(ValidationError):
        AskRequest(question="")


def test_k_is_bounded():
    with pytest.raises(ValidationError):
        AskRequest(question="hello", k=0)
    with pytest.raises(ValidationError):
        AskRequest(question="hello", k=500)


def test_chunk_size_outside_the_assignment_range_is_rejected():
    # 300-500 tokens. A config that says 4000 is a typo, not a design choice.
    with pytest.raises(ValidationError):
        ChunkingParams(chunk_size_tokens=4000)
    with pytest.raises(ValidationError):
        ChunkingParams(chunk_size_tokens=50)


def test_overlap_ratio_outside_the_assignment_range_is_rejected():
    with pytest.raises(ValidationError):
        ChunkingParams(overlap_ratio=0.0)
    with pytest.raises(ValidationError):
        ChunkingParams(overlap_ratio=0.5)


def test_chunking_params_derive_overlap_and_stride():
    p = ChunkingParams(chunk_size_tokens=400, overlap_ratio=0.15)
    assert p.overlap_tokens == 60
    assert p.stride_tokens == 340


def test_similarity_outside_minus_one_to_one_is_rejected():
    # A number above 1.0 means un-normalised vectors and a dot product wearing a
    # cosine-similarity label.
    with pytest.raises(ValidationError):
        RetrievedChunk(
            chunk_id=1, document_id=1, text="x", similarity=7.3, rank=1
        )


def test_rank_starts_at_one():
    with pytest.raises(ValidationError):
        RetrievedChunk(chunk_id=1, document_id=1, text="x", similarity=0.5, rank=0)


def test_embed_request_requires_at_least_one_document():
    with pytest.raises(ValidationError):
        EmbedRequest(documents=[])


def test_embed_request_corpus_is_one_of_two_values():
    with pytest.raises(ValidationError):
        EmbedRequest(
            documents=[{"title": "t", "source": "s", "text": "body"}],
            corpus="both",
        )


def test_retrieved_chunk_round_trips():
    r = RetrievedChunk(
        chunk_id=12,
        document_id=3,
        document_title="Support rota",
        text="Requests after 16:00 are queued for the next working day.",
        similarity=0.81,
        rank=1,
    )
    assert RetrievedChunk(**r.model_dump()) == r
