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

THE ONE IDEA THIS FILE IS BUILT AROUND
--------------------------------------
An entity is a **character span plus a type**: ``(start_char, end_char, type)``
measured against the original string the user sent, not against tokens, not
against subwords, not against a lowercased copy. Everything downstream — the
inline highlighting, the review queue, the entity-level scores — breaks the
moment a span stops indexing into the original text. If you remember one thing
from this template, make it that.

You MAY add fields. If your extractor reports, say, a second-best type per
entity, add it to ``Entity`` and render it in the UI. Do NOT rename or delete
the existing fields — the tests and the grading rubric reference them by name.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


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

    # "model_type" and "model_version" are our field names, not pydantic's. This
    # line stops pydantic warning about the "model_" prefix on every import.
    model_config = ConfigDict(protected_namespaces=())

    service: str
    git_sha: str = Field(description="Short commit SHA of the running build")
    model_version: str = Field(
        description="Free-text version of the NER configuration, e.g. 'entityfinder-v2'"
    )


# ---------------------------------------------------------------------------
# The core object: one predicted entity
# ---------------------------------------------------------------------------
ModelName = Literal["transformer", "crf"]

#: A scored-as-correct-or-not entity, reduced to the three things entity-level
#: scoring compares: where it starts, where it ends, and what type it is. The
#: surface string and the confidence are deliberately NOT part of it — two
#: predictions that agree on all three of these are the same entity.
Span = Tuple[int, int, str]


class Entity(BaseModel):
    """One predicted entity, located by character offsets into the input text.

    ``text[start_char:end_char]`` must equal ``text`` on this model. If it does
    not, your offsets are wrong, and every downstream artefact — the highlight
    in the UI, the reviewer's correction, the score in your report — is wrong
    with them. There is a contract test for exactly this.
    """

    text: str = Field(description="Surface string, exactly text[start_char:end_char]")
    start_char: int = Field(ge=0, description="Inclusive start offset in the ORIGINAL text")
    end_char: int = Field(gt=0, description="Exclusive end offset in the ORIGINAL text")
    entity_type: str = Field(description='e.g. "PER", "ORG", "LOC", "MISC"')
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model score for this span in [0, 1]. This is what the review "
        "queue thresholds on, so a placeholder 1.0 makes the queue useless.",
    )
    entity_id: Optional[int] = Field(
        default=None,
        description="Row id in the Supabase entities table, filled in by the API "
        "after logging. The Review Queue needs it to write a decision back.",
    )
    context: Optional[str] = Field(
        default=None,
        description="Optional snippet of surrounding text, for the reviewer. See the "
        "privacy note in db/migrations/001_init.sql before you turn this on.",
    )


class TokenPrediction(BaseModel):
    """One token, its BIO tag, its character offsets, and the model's score.

    This is the intermediate representation between the model and the entities:
    a token-level view that still knows where it came from in the original
    string. ``decode_bio_spans()`` turns a list of these into ``Entity`` objects.

    ``start_char`` / ``end_char`` come from the tokenizer's offset mapping — from
    ``return_offsets_mapping=True``, not from counting characters yourself and
    not from ``str.find()`` on the token, both of which fall apart the first time
    a token repeats in the sentence.
    """

    token: str
    tag: str = Field(description='BIO tag: "O", "B-PER", "I-ORG", ...')
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0, description="Score for THIS token's tag")


# ---------------------------------------------------------------------------
# POST /extract
# ---------------------------------------------------------------------------
class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, description="Raw document text from the analyst")
    model: ModelName = Field(
        default="transformer",
        description='Which extractor to run: "transformer" or "crf". The UI calls '
        "this endpoint twice with different values to build the comparison tab.",
    )


class ExtractResponse(BaseModel):
    """Everything the Extract Entities tab needs to draw the highlighted document."""

    # "model_type" and "model_version" are our field names, not pydantic's. This
    # line stops pydantic warning about the "model_" prefix on every import.
    model_config = ConfigDict(protected_namespaces=())

    text: str = Field(
        description="The input, echoed so the UI can highlight without re-sending it. "
        "Echoing is not storing — see text_sha256 for what the database gets."
    )
    text_sha256: str = Field(description="Hash of the input; this is what is persisted")
    model: ModelName
    model_version: str
    entities: List[Entity]
    entity_count: int
    latency_ms: Optional[int] = None
    extraction_id: Optional[int] = Field(
        default=None, description="Row id in the extractions table, if logging succeeded"
    )


# ---------------------------------------------------------------------------
# GET /review_queue  and  POST /review
# ---------------------------------------------------------------------------
class ReviewQueueItem(BaseModel):
    """One low-confidence prediction queued for a human.

    Note what a reviewer gets: the entity, its span, its type, its score, and
    the run that produced it — but not the document, because the document was
    never stored. Whether that is enough context to review from is a real design
    question and your report should answer it rather than dodge it.
    """

    # "model_type" and "model_version" are our field names, not pydantic's. This
    # line stops pydantic warning about the "model_" prefix on every import.
    model_config = ConfigDict(protected_namespaces=())

    entity_id: int
    extraction_id: int
    text: str
    start_char: int
    end_char: int
    entity_type: str
    confidence: float
    model: Optional[str] = None
    model_version: Optional[str] = None
    context: Optional[str] = None
    created_at: Optional[str] = None


class ReviewQueueResponse(BaseModel):
    threshold: float = Field(
        description="The confidence cutoff used for this query. It is your team's "
        "number, not a constant of nature — justify it in the report."
    )
    count: int
    items: List[ReviewQueueItem]


ReviewDecision = Literal["accept", "correct", "reject"]


class ReviewRequest(BaseModel):
    """A reviewer's verdict on one predicted entity.

    Three decisions, and they are not the same thing:

    * ``accept``  — the prediction is right as it stands.
    * ``correct`` — there is an entity here, but the type and/or the boundary is
      wrong. Send the fix in the ``corrected_*`` fields.
    * ``reject``  — there is no entity here at all. A false positive.

    Collapsing ``reject`` into ``correct`` loses the distinction between "you
    mislabelled something real" and "you hallucinated an entity", and those two
    errors call for completely different fixes.
    """

    entity_id: int
    decision: ReviewDecision
    reviewer_id: str = Field(
        min_length=1,
        description="Who reviewed it. An opaque id or initials is fine; this is an "
        "audit trail, not a personnel file.",
    )
    corrected_type: Optional[str] = None
    corrected_start_char: Optional[int] = Field(default=None, ge=0)
    corrected_end_char: Optional[int] = Field(default=None, ge=0)
    note: Optional[str] = Field(
        default=None, description="Free text — why the reviewer decided what they did"
    )

    @model_validator(mode="after")
    def _correction_must_actually_correct_something(self) -> "ReviewRequest":
        if self.decision == "correct":
            if (
                self.corrected_type is None
                and self.corrected_start_char is None
                and self.corrected_end_char is None
            ):
                raise ValueError(
                    'decision="correct" requires at least one of corrected_type, '
                    "corrected_start_char, corrected_end_char"
                )
        if (
            self.corrected_start_char is not None
            and self.corrected_end_char is not None
            and self.corrected_end_char <= self.corrected_start_char
        ):
            raise ValueError("corrected_end_char must be greater than corrected_start_char")
        return self


class ReviewResponse(BaseModel):
    """The stored decision, with the ORIGINAL prediction echoed back.

    The echo is not decoration. It is the proof that the write preserved what
    the model said instead of overwriting it. If ``original_*`` ever comes back
    equal to the correction, someone has updated the entities row in place and
    the audit trail is gone.
    """

    review_id: int
    entity_id: int
    decision: ReviewDecision
    reviewer_id: str
    original_type: str
    original_start_char: int
    original_end_char: int
    original_confidence: float
    corrected_type: Optional[str] = None
    corrected_start_char: Optional[int] = None
    corrected_end_char: Optional[int] = None
    note: Optional[str] = None
    created_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Dataset profile and training runs — GET /runs
# ---------------------------------------------------------------------------
class DatasetProfile(BaseModel):
    """What you found when you looked at the corpus, instead of assuming.

    The label distribution is the field students skip and then regret. NER data
    is overwhelmingly ``O``; if you do not know that ratio for your own splits,
    you cannot say anything sensible about why token-level accuracy looks so good.
    """

    name: str = Field(description='e.g. "conll2003"')
    splits: Dict[str, int] = Field(
        description='Sentence counts per split, e.g. {"train": ..., "validation": ..., "test": ...}'
    )
    entity_types: List[str] = Field(description='e.g. ["PER", "ORG", "LOC", "MISC"]')
    label_distribution: Dict[str, int] = Field(
        description='Token counts per BIO tag, INCLUDING "O". Report the O share.'
    )
    tagging_scheme: Optional[str] = Field(
        default=None, description='"IOB1" or "IOB2" — check, do not assume'
    )
    notes: Optional[str] = None


class EntityScores(BaseModel):
    """Entity-level precision, recall, and F1. Not token-level. Not accuracy.

    A prediction counts as a true positive only when its start offset, its end
    offset, AND its type all match a gold entity. "New York Stock Exchange"
    tagged as ORG but starting one word late is a false positive and a false
    negative at once — it scores worse than saying nothing.
    """

    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    support: int = Field(description="Number of gold entities scored against")
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    per_type: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description='Per-entity-type breakdown, e.g. {"ORG": {"precision": ..., '
        '"recall": ..., "f1": ..., "support": ...}}. The aggregate number hides '
        "which type you are actually bad at.",
    )


class TrainingRun(BaseModel):
    """One training run of one model, as it goes into the runs table."""

    # "model_type" and "model_version" are our field names, not pydantic's. This
    # line stops pydantic warning about the "model_" prefix on every import.
    model_config = ConfigDict(protected_namespaces=())

    model_type: ModelName
    dataset: str
    hyperparameters: Dict[str, object] = Field(
        default_factory=dict,
        description="Everything needed to re-run this: learning rate, epochs, seed, "
        "checkpoint name for the transformer; c1/c2 and the feature set for the CRF.",
    )
    features: List[str] = Field(
        default_factory=list,
        description="CRF feature names. Empty for the transformer, which learns its "
        "own — that contrast is half the point of the comparison.",
    )
    metrics: Optional[EntityScores] = None
    model_version: str
    notes: Optional[str] = None


class Run(BaseModel):
    """One logged training run, as it comes back out of Supabase."""

    # "model_type" and "model_version" are our field names, not pydantic's. This
    # line stops pydantic warning about the "model_" prefix on every import.
    model_config = ConfigDict(protected_namespaces=())

    id: int
    model_type: str = Field(description='"crf" or "transformer"')
    dataset: str
    config: dict = Field(description="Hyperparameters and feature set, as JSON")
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    metrics: Optional[dict] = Field(
        default=None, description="Full EntityScores payload including per_type, as JSON"
    )
    model_version: str
    notes: Optional[str] = None
    created_at: Optional[str] = None


class RunsResponse(BaseModel):
    runs: List[Run]
