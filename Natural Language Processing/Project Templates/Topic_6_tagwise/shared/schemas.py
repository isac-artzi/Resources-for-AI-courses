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

You MAY add fields. If your evaluation reports per-tag precision, add it to
``Run.metrics`` and render it in the UI. Do NOT rename or delete the existing
fields — the tests and the grading rubric reference them by name.

TWO KINDS OF RECORD, AND THE DIFFERENCE MATTERS
-----------------------------------------------
``Run`` describes *building a tagger*: one lookup table built, or one transformer
fine-tuned, with the hyperparameters and the evaluation numbers that came out.
``Tagging`` describes *answering one request*: someone sent a sentence, a model
produced a tag sequence. If you log only runs, the History tab has nothing to
show — a service that records how it was trained but never records what it
answered has no audit trail at all.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# The two taggers this service exposes. "baseline" is the most-frequent-tag
# lookup with a rule-based fallback; "transformer" is your fine-tuned token
# classifier. Anything else is a 422 before it ever reaches your code.
ModelName = Literal["baseline", "transformer"]


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
        description="Free-text version of the tagger configuration, e.g. 'tagwise-v2'"
    )


# ---------------------------------------------------------------------------
# POST /tag
# ---------------------------------------------------------------------------
class TaggedToken(BaseModel):
    """One word and the tag a model gave it.

    ``used_fallback`` is the field that makes the baseline honest. When the
    lookup table has never seen a word, the rule-based fallback guesses, and the
    UI colours that token differently. Almost all of the baseline's remaining
    error lives in those tokens, so hiding them would hide the finding.
    """

    token: str = Field(description="The word as it appeared in the input")
    tag: str = Field(description="Predicted tag, e.g. a universal tag like NOUN or VERB")
    confidence: Optional[float] = Field(
        default=None,
        description="Model probability for the chosen tag, in [0, 1]. The lookup "
        "baseline has no probability of its own — leave it None rather than "
        "inventing a 1.0 that will mislead whoever reads the UI.",
    )
    used_fallback: bool = Field(
        default=False,
        description="True when the baseline had never seen this word and the "
        "unknown-word rules produced the tag. Always False for the transformer.",
    )


class TagRequest(BaseModel):
    sentence: str = Field(min_length=1, description="One sentence of raw text")
    model: ModelName = Field(
        default="baseline",
        description='Which tagger to use: "baseline" or "transformer". The '
        "default is the baseline because it needs no downloaded weights, so a "
        "fresh fork can answer a request as soon as you implement the lookup.",
    )


class TagResponse(BaseModel):
    """Everything the 'Tag a Sentence' tab needs to draw the coloured sentence."""

    sentence: str
    model: ModelName
    tokens: List[TaggedToken]
    tag_sequence: List[str] = Field(
        description="The tags alone, in order — the compact form that gets logged"
    )
    unknown_count: int = Field(
        default=0, description="How many tokens the baseline had to fall back on"
    )
    model_version: str
    tagging_id: Optional[int] = Field(
        default=None, description="Row id in the Supabase taggings table, if logging succeeded"
    )


# ---------------------------------------------------------------------------
# GET /runs
# ---------------------------------------------------------------------------
class Run(BaseModel):
    """One tagger build — the 'Baseline vs. Transformer' tab reads these.

    Both taggers write a row here. The lookup table is not trained by gradient
    descent, but it is still built from a specific training split with specific
    choices (lowercasing, tie-breaking, fallback rules), and those choices belong
    in ``hyperparameters`` exactly like a learning rate does. If you cannot say
    which build produced a number, you cannot compare two numbers.
    """

    id: int
    model: str = Field(description='"baseline" or "transformer"')
    tagset: str = Field(
        description='Which label set, e.g. "UPOS" for the 17 universal tags. '
        "Two accuracy numbers computed over different tag sets are not comparable."
    )
    hyperparameters: dict = Field(
        default_factory=dict,
        description="Everything needed to rebuild this tagger: treebank name and "
        "revision, split sizes, learning rate, epochs, base checkpoint, and the "
        "baseline's casing and tie-breaking rules.",
    )
    accuracy: Optional[float] = Field(
        default=None, description="Token-level accuracy on the held-out split, in [0, 1]"
    )
    macro_f1: Optional[float] = Field(
        default=None,
        description="Unweighted mean of per-tag F1. Rare tags count as much as "
        "common ones here, which is the whole reason to report it next to accuracy.",
    )
    metrics: dict = Field(
        default_factory=dict,
        description='Anything else worth keeping, including the confusion matrix '
        'as {"labels": [...], "matrix": [[...]]}. The comparison tab renders it '
        "from here, which is why there is no separate evaluation endpoint.",
    )
    model_version: str
    notes: Optional[str] = None
    created_at: Optional[str] = None


class RunsResponse(BaseModel):
    runs: List[Run]


# ---------------------------------------------------------------------------
# The taggings table (read by the UI directly; no endpoint serves it)
# ---------------------------------------------------------------------------
class Tagging(BaseModel):
    """One served request, as it is stored.

    The UI's History tab queries this table straight from Postgres with the anon
    key, so nothing in the API returns this model today. It is written down here
    anyway: the table's shape is part of the contract between the tiers, and
    typing it means the UI and the API cannot disagree about what a row is.
    """

    id: int
    sentence_sha256: str = Field(
        description="Hash of the input sentence, never the sentence itself. "
        "A tagged sentence is still someone's text."
    )
    token_count: int
    tag_sequence: List[str] = Field(description="Predicted tags, in order")
    model: str
    model_version: str
    unknown_count: int = 0
    created_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Evaluation shapes (used inside api/nlp.py, not served on their own)
# ---------------------------------------------------------------------------
class ConfusionMatrix(BaseModel):
    """Counts of gold tag (row) against predicted tag (column).

    ``labels`` fixes the order of both axes, so ``matrix[i][j]`` is "gold
    labels[i], predicted labels[j]". Keep it square and keep the same label list
    on both axes, or the diagonal stops meaning "correct".

    Seventeen universal tags give you a 17x17 grid that a reader can scan in a
    few seconds and point at: DET confused with PRON, NOUN with VERB, ADJ with
    ADV. A 45-tag set gives you 2,025 cells, and nobody reads that.
    """

    labels: List[str]
    matrix: List[List[int]]


class EvalReport(BaseModel):
    """The numbers one tagger scored on one split. Goes into a Run row."""

    model: str
    accuracy: float
    macro_f1: float
    per_tag_f1: Dict[str, float] = Field(default_factory=dict)
    confusion: Optional[ConfusionMatrix] = None
    n_tokens: int = 0
