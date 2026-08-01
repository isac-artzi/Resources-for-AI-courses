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

You MAY add fields. If your tokenizer comparison wants to report, say, the
average token length, add it to ``TokenizeResponse`` and render it in the UI.
Do NOT rename or delete the existing fields — the tests and the grading rubric
reference them by name.
"""
from __future__ import annotations

from typing import Dict, List, Optional

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
        description="Free-text version of the NLP configuration, e.g. 'wordpiece-v2'"
    )


# ---------------------------------------------------------------------------
# POST /preprocess
# ---------------------------------------------------------------------------
class PreprocessOptions(BaseModel):
    """Which classical preprocessing steps to apply, in this order.

    The order matters and is deliberately fixed: lowercase → strip punctuation →
    remove stop words → stem OR lemmatize. Applying stemming before stop-word
    removal, for example, changes which words match the stop list. Part of your
    report is explaining what each toggle did to the token count.
    """

    lowercase: bool = True
    strip_punctuation: bool = True
    remove_stopwords: bool = True
    remove_digits: bool = False
    stem: bool = False
    lemmatize: bool = True


class PreprocessRequest(BaseModel):
    text: str = Field(min_length=1, description="Raw input text from the analyst")
    options: PreprocessOptions = PreprocessOptions()


class PreprocessResponse(BaseModel):
    """Everything the 'Clean & Tokenize' tab needs to draw the before/after view."""

    original_text: str
    cleaned_text: str
    original_tokens: List[str] = Field(
        description="Whitespace/word tokens of the ORIGINAL text — the 'before' side"
    )
    cleaned_tokens: List[str] = Field(description="Tokens surviving preprocessing")
    steps_applied: List[str] = Field(
        description="Human-readable step names, in the order they ran. "
        "This is what makes the pipeline auditable — do not skip it."
    )
    token_count_before: int
    token_count_after: int
    run_id: Optional[int] = Field(
        default=None, description="Row id in the Supabase runs table, if logging succeeded"
    )


# ---------------------------------------------------------------------------
# POST /tokenize
# ---------------------------------------------------------------------------
class TokenizerResult(BaseModel):
    """One subword tokenizer's view of the same text."""

    tokenizer_name: str = Field(description='Hugging Face id, e.g. "bert-base-uncased"')
    algorithm: str = Field(description='"WordPiece", "BPE", "Unigram", ...')
    tokens: List[str] = Field(description="Subword pieces, in order, including markers")
    token_ids: List[int]
    token_count: int
    unknown_count: int = Field(
        description="How many pieces mapped to the tokenizer's unknown token"
    )
    oov_rate: float = Field(
        description="unknown_count / token_count, in [0, 1]. Report it — this is "
        "the number that shows why subword tokenization exists."
    )
    vocab_size: int


class TokenizeRequest(BaseModel):
    text: str = Field(min_length=1)
    tokenizers: List[str] = Field(
        default=["bert-base-uncased", "gpt2"],
        min_length=2,
        description="At least two Hugging Face tokenizers, so the Compare tab has "
        "something to compare. The assignment requires two different algorithms "
        "(e.g. WordPiece and BPE), not two checkpoints of the same one.",
    )


class TokenizeResponse(BaseModel):
    text: str
    results: List[TokenizerResult]
    vocabulary_overlap: Dict[str, float] = Field(
        default_factory=dict,
        description='Pairwise Jaccard overlap of the produced piece sets, keyed '
        '"tokenizer_a|tokenizer_b". Empty dict is acceptable for a single result.',
    )
    run_id: Optional[int] = None


# ---------------------------------------------------------------------------
# GET /runs
# ---------------------------------------------------------------------------
class Run(BaseModel):
    """One logged pass through the service — the History tab reads these."""

    id: int
    kind: str = Field(description='"preprocess" or "tokenize"')
    text_sha256: str = Field(
        description="Hash of the input, never the input itself. The corpus may be "
        "someone's private data; you log what you need to reproduce, not the text."
    )
    config: dict = Field(description="The options/tokenizer list used, as JSON")
    token_count_before: Optional[int] = None
    token_count_after: Optional[int] = None
    oov_rate: Optional[float] = None
    model_version: str
    created_at: Optional[str] = None


class RunsResponse(BaseModel):
    runs: List[Run]
