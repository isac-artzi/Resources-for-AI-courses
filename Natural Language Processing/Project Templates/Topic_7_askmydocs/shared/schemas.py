"""Pydantic request/response models — the contract between the two clouds.

WHY THIS FILE EXISTS
--------------------
The UI (Streamlit Cloud) and the API (Render.com) are separate deployments that
only ever talk over HTTPS + JSON. This module is the single written-down
description of that JSON. Both tiers import it, so a change here is a change on
both sides at once and you cannot drift.

WHAT YOU SHOULD DO WITH IT
--------------------------
Read it first, before you write any NLP code. Every stub in ``api/nlp.py`` is
typed in terms of these models, so the shapes below tell you exactly what your
functions have to produce.

You MAY add fields. If your retrieval step wants to report, say, a reranker
score alongside the cosine similarity, add it to ``RetrievedChunk`` and render
it in the Retrieval Audit tab. Do NOT rename or delete the existing fields — the
tests and the grading rubric reference them by name.

ONE THING TO NOTICE BEFORE YOU READ ON
--------------------------------------
``AskResponse`` carries both the answer AND the passages that were retrieved,
with their similarity scores. That is not decoration. An answer that ignored a
perfectly good retrieved passage and an answer that had nothing useful to work
with look identical from the outside — the same wrong sentence. The only way to
tell them apart is to log what came back from the vector store next to what the
generator produced. Half of this assignment is that audit trail.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Infrastructure responses (already implemented — you should not need to touch)
# ---------------------------------------------------------------------------
class Health(BaseModel):
    """GET /healthz — is the service up, and can it reach the database?"""

    status: str = Field(description='"ok" when the process is serving requests')
    database: str = Field(description='"ok", "unreachable", or "not_configured"')


class Version(BaseModel):
    """GET /version — which build is live right now?

    Graders use this to confirm the URL in your README is the code you claim.
    """

    service: str
    git_sha: str = Field(description="Short commit SHA of the running build")
    model_version: str = Field(
        description="Free-text version of the RAG configuration, e.g. 'askmydocs-v2'"
    )


# ---------------------------------------------------------------------------
# Chunking parameters — shared by POST /embed and recorded on every chunk row
# ---------------------------------------------------------------------------
class ChunkingParams(BaseModel):
    """How a document is cut into passages, and the record of that decision.

    The bounds below are the assignment's, and they are enforced here rather
    than left to your good intentions: 300–500 tokens per passage, 10–20 percent
    overlap. Pydantic rejects anything outside that range before it reaches your
    code, which is deliberate — "I meant to use 400 but the config said 4000" is
    a bug you find three days later, when every embedding is already written.

    Why these numbers matter, in one paragraph you can reuse in your report:
    a passage that is too short loses the context that makes it answer the
    question; a passage that is too long dilutes its own embedding, because one
    vector now has to represent five unrelated paragraphs and ends up close to
    nothing in particular. The overlap exists so that a fact sitting on a chunk
    boundary appears whole in at least one chunk.
    """

    chunk_size_tokens: int = Field(
        default=400,
        ge=300,
        le=500,
        description="Target passage length in tokens, as counted by tokenizer_name",
    )
    overlap_ratio: float = Field(
        default=0.15,
        ge=0.10,
        le=0.20,
        description="Fraction of chunk_size_tokens repeated between neighbours",
    )
    tokenizer_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="The tokenizer the token counts are measured with. Record it: "
        "'400 tokens' means nothing until you say whose tokens.",
    )

    @property
    def overlap_tokens(self) -> int:
        """Overlap expressed in tokens. Rounded, never larger than the chunk."""
        return min(self.chunk_size_tokens - 1, round(self.chunk_size_tokens * self.overlap_ratio))

    @property
    def stride_tokens(self) -> int:
        """How far the window advances between chunks. Always >= 1."""
        return max(1, self.chunk_size_tokens - self.overlap_tokens)


# ---------------------------------------------------------------------------
# POST /embed — ingest documents into a corpus
# ---------------------------------------------------------------------------
class DocumentIn(BaseModel):
    """One document on its way into the store."""

    title: str = Field(min_length=1, description="Human-readable name, shown as a citation")
    source: str = Field(
        min_length=1,
        description="Where it came from: a URL, a filename, a dataset id. The "
        "product brief asks you to document the collection's source, and this "
        "is where that promise is kept per document.",
    )
    text: str = Field(min_length=1, description="Full plain text of the document")


class EmbedRequest(BaseModel):
    """Ingest a batch of documents, chunk them, embed them, store them.

    ``corpus`` is the field that keeps this assignment honest. A document is
    either part of the RETRIEVAL collection (the thing you search at query time)
    or part of the FINE-TUNING corpus (the thing you adapted the generator on).
    Never both. The API refuses a document whose content hash already exists in
    the other corpus, and the database has a unique index that refuses it again
    if the API check is ever bypassed.
    """

    documents: List[DocumentIn] = Field(min_length=1)
    chunking: ChunkingParams = ChunkingParams()
    corpus: Literal["retrieval", "finetune"] = "retrieval"
    finetune_sample: List[str] = Field(
        default_factory=list,
        description="Optional. Raw texts from your fine-tuning corpus. When you "
        "send them, the service runs the near-duplicate disjointness check "
        "before writing anything, and refuses the ingest with a 409 if the two "
        "collections overlap. Exact-hash collisions are always checked; this "
        "field is how you also catch a paraphrase or a re-formatted copy.",
    )


class ChunkRecord(BaseModel):
    """One stored passage, as reported back to the caller."""

    chunk_id: Optional[int] = Field(
        default=None, description="Row id in the chunks table, if the write succeeded"
    )
    document_id: Optional[int] = None
    ordinal: int = Field(ge=0, description="0-based position of this chunk within its document")
    text: str
    token_count: int
    start_token: int = Field(ge=0, description="Inclusive token offset into the document")
    end_token: int = Field(ge=0, description="Exclusive token offset into the document")


class EmbedResponse(BaseModel):
    """What the ingest actually did — every number here belongs in your report."""

    corpus: str
    document_count: int
    chunk_count: int
    embedding_model: str = Field(description="Exact model id used to produce the vectors")
    embedding_dim: int = Field(
        description="Length of each stored vector. It MUST equal the dimension of "
        "the pgvector column, and it must equal the dimension you query with."
    )
    chunking: ChunkingParams
    chunk_ids: List[int] = Field(default_factory=list)
    stored: bool = Field(description="False when Supabase was unreachable and only chunking ran")


# ---------------------------------------------------------------------------
# POST /ask — the product
# ---------------------------------------------------------------------------
class RetrievedChunk(BaseModel):
    """One passage the retriever returned, with the score that got it there."""

    chunk_id: int
    document_id: int
    document_title: Optional[str] = None
    text: str = Field(description="The passage itself — this is what the grader reads")
    similarity: float = Field(
        ge=-1.0,
        le=1.0,
        description="Cosine similarity to the query embedding, in [-1, 1]. If your "
        "numbers land outside that range you are not computing cosine similarity; "
        "you are computing a dot product over vectors you forgot to normalise.",
    )
    rank: int = Field(ge=1, description="1 = closest. Rank 1 through k, no gaps.")


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="How many passages to retrieve. Start at 5 and justify any change.",
    )
    use_retrieval: bool = Field(
        default=True,
        description="False runs the generator with no context at all. This is the "
        "control condition for the with/without comparison the assignment asks "
        "for — same question, same model, same decoding settings, no passages.",
    )


class AskResponse(BaseModel):
    """The answer plus everything needed to audit how it was produced."""

    question: str
    answer: str
    retrieval_used: bool
    k: int
    retrieved: List[RetrievedChunk] = Field(
        default_factory=list,
        description="Empty when retrieval_used is False. Otherwise up to k passages, "
        "rank 1 first.",
    )
    cited_chunk_ids: List[int] = Field(
        default_factory=list,
        description="The subset of retrieved chunk ids the answer actually cites. "
        "The gap between 'retrieved' and 'cited' is the interesting one: a passage "
        "that was retrieved and then ignored is a different failure from a passage "
        "that was never found, and you can only tell them apart if you log both.",
    )
    generator_model: str = Field(description="Exact generator id or inference endpoint used")
    embedding_model: Optional[str] = Field(
        default=None, description="Model used to embed the query. None when retrieval was off."
    )
    prompt_token_count: Optional[int] = Field(
        default=None,
        description="Tokens in the final prompt. Watch this number: it is how you "
        "find out you silently truncated the context window.",
    )
    query_id: Optional[int] = Field(
        default=None, description="Row id in the Supabase queries table, if logging succeeded"
    )


# ---------------------------------------------------------------------------
# GET /sources
# ---------------------------------------------------------------------------
class SourceSummary(BaseModel):
    """One document in the store, as the Sources view lists it."""

    document_id: int
    title: str
    source: str
    corpus: str = Field(description='"retrieval" or "finetune"')
    token_count: Optional[int] = None
    chunk_count: int = 0


class SourcesResponse(BaseModel):
    sources: List[SourceSummary]
    document_count: int = 0
    chunk_count: int = 0


# ---------------------------------------------------------------------------
# Evaluation artefacts — produced by api/nlp.py, quoted in your report
# ---------------------------------------------------------------------------
class PerplexityReport(BaseModel):
    """Perplexity of the generator on a held-out split of the FINE-TUNING corpus.

    Not the retrieval collection. Perplexity measured on documents the model was
    trained on is a measure of memorisation, and a memorising model reports a
    beautiful number while being useless at everything the product does.
    """

    model_name: str
    split: str = Field(default="held_out", description='"held_out" or "train"')
    document_count: int
    token_count: int = Field(description="Total tokens scored — perplexity per token")
    perplexity: float = Field(gt=0.0, description="exp(mean negative log-likelihood per token)")


class OverlapPair(BaseModel):
    """One suspicious pairing found by the disjointness check."""

    finetune_index: int
    retrieval_index: int
    jaccard: float = Field(ge=0.0, le=1.0)
    reason: str = Field(description='"exact_hash" or "shingle_overlap"')


class DisjointnessReport(BaseModel):
    """Evidence that the fine-tuning corpus and the retrieval collection are separate.

    Paste this into your report. "We kept them separate" is a claim; this is the
    check that backs it.
    """

    finetune_document_count: int
    retrieval_document_count: int
    exact_duplicate_count: int = 0
    max_shingle_jaccard: float = Field(default=0.0, ge=0.0, le=1.0)
    overlapping_pairs: List[OverlapPair] = Field(default_factory=list)
    disjoint: bool = Field(description="True when nothing crossed the threshold")


class ComparisonRow(BaseModel):
    """One question answered twice — with retrieval and without.

    Build at least ten of these. Three of them have to be cases where the
    retrieved passage stopped the model inventing something, and you have to
    quote the passage that did it.
    """

    question: str
    answer_with_retrieval: str
    answer_without_retrieval: str
    cited_chunk_ids: List[int] = Field(default_factory=list)
    grounding_passage: Optional[str] = Field(
        default=None, description="The passage you claim prevented the hallucination"
    )
    notes: Optional[str] = None


class ComparisonReport(BaseModel):
    rows: List[ComparisonRow] = Field(default_factory=list)
    counts: Dict[str, int] = Field(
        default_factory=dict,
        description='Free-form tallies, e.g. {"hallucination_prevented": 3}',
    )
