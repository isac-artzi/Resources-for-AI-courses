"""Pydantic request/response models — the contract between the two clouds.

WHY THIS FILE EXISTS
--------------------
The UI (Streamlit Cloud) and the API (Render.com) are separate deployments that
only ever talk over HTTPS + JSON. This module is the single written-down
description of that JSON. Both tiers import it, so a change here is a change on
both sides at once and you cannot drift.

WHAT YOU SHOULD DO WITH IT
--------------------------
Read it first, before you write any classification code. Every stub in
``api/nlp.py`` is typed in terms of these models, so the shapes below tell you
exactly what your functions have to produce.

THE ONE RULE THAT IS ENFORCED HERE, NOT LEFT TO YOU
---------------------------------------------------
This product is **single-label, binary**: every example carries exactly one
label, and that label is one of exactly two values. ``LabelSchema`` below
refuses to construct otherwise, on purpose. Multilabel (several labels per
example) and multiclass (more than two label values) change the loss function,
the metric definitions, and the meaning of "the probability" — they are
addressed in later courses, and quietly half-supporting them here is how a
project ends up reporting a number nobody can interpret.

You MAY add fields. If your comparison wants to report, say, inference latency
per model, add it and render it in the UI. Do NOT rename or delete the existing
fields — the tests and the grading rubric reference them by name.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

# The only two model kinds this service serves. The classical baseline exists so
# that "the transformer is better" is a measurement rather than an assumption.
ModelKind = Literal["baseline", "transformer"]


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
        description="Free-text version of the served model configuration, "
        "e.g. 'distilbert-ft-v2'"
    )


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------
class LabeledExample(BaseModel):
    """One row of the training corpus: a piece of text and its single label.

    ``label`` is a string, not a list. That is the single-label rule expressed in
    the type system. If your source data has a list of labels per row, it is a
    multilabel dataset and it is out of scope for this product — pick a different
    corpus rather than collapsing the list to its first element, which silently
    throws away supervision and makes your metrics meaningless.
    """

    text: str = Field(min_length=1)
    label: str = Field(min_length=1)


class LabelSchema(BaseModel):
    """GET /schema — what this classifier predicts, and on what data it learned.

    This is the model's self-description. A caller who has never seen your repo
    should be able to hit /schema and know what the two labels mean, which one
    the reported precision and recall are computed for, and how imbalanced the
    training data was.
    """

    task: str = Field(
        default="single-label binary text classification",
        description="Free text, but it should say single-label and binary.",
    )
    labels: List[str] = Field(
        min_length=2,
        max_length=2,
        description="Exactly two label values, in a stable order. Not a "
        "suggestion — the validator below enforces it.",
    )
    positive_label: str = Field(
        description="Which of the two labels precision, recall and F1 are "
        "computed for. There is no such thing as 'the F1' without this."
    )
    label_definitions: Dict[str, str] = Field(
        default_factory=dict,
        description="One sentence per label describing what an annotator had to "
        "see to assign it. The Build Steps ask for this explicitly.",
    )
    class_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Rows per label in the training corpus. This is the number "
        "that tells a reader whether accuracy means anything.",
    )
    dataset_name: str = Field(default="", description="Human-readable corpus name")
    dataset_source: str = Field(
        default="", description="Public URL or citation for the corpus"
    )
    n_rows: int = Field(default=0, ge=0, description="Total labeled rows")

    @model_validator(mode="after")
    def _enforce_single_label_binary(self) -> "LabelSchema":
        # Two DISTINCT values. ["spam", "spam"] passes min_length/max_length and
        # is obviously wrong, so check distinctness explicitly.
        if len(set(self.labels)) != 2:
            raise ValueError(
                f"labels must be exactly two distinct values, got {self.labels!r}. "
                "This product is single-label binary by design: multiclass and "
                "multilabel are out of scope."
            )
        if self.positive_label not in self.labels:
            raise ValueError(
                f"positive_label {self.positive_label!r} is not one of {self.labels!r}. "
                "Precision and recall are computed for one named class; if that "
                "class is not in the label set, the numbers are undefined."
            )
        if self.class_counts and set(self.class_counts) != set(self.labels):
            raise ValueError(
                f"class_counts keys {sorted(self.class_counts)} do not match "
                f"labels {sorted(self.labels)}. A count for a label you do not "
                "predict means your loader and your schema disagree about the data."
            )
        return self

    def minority_share(self) -> Optional[float]:
        """Fraction of the corpus in the smaller class, or None if unknown.

        IMPLEMENTED FOR YOU because it is the first thing you should look at.
        If this returns 0.04, a model that always answers with the majority label
        scores 96% accuracy while being useless, and any report that leads with
        accuracy is misleading its reader.
        """
        if not self.class_counts:
            return None
        total = sum(self.class_counts.values())
        if total <= 0:
            return None
        return min(self.class_counts.values()) / total


# ---------------------------------------------------------------------------
# POST /predict and POST /predict_batch
# ---------------------------------------------------------------------------
class PredictionResult(BaseModel):
    """One served prediction. This is also, field for field, what gets logged."""

    label: str = Field(description="The predicted label — one of LabelSchema.labels")
    probability: float = Field(
        ge=0.0,
        le=1.0,
        description="CALIBRATED probability of the predicted label. Not the raw "
        "softmax output and not decision_function(). See nlp.calibrate().",
    )
    model_kind: ModelKind
    model_version: str = Field(
        description="Identifies the ARTIFACT that answered, e.g. "
        "'distilbert-ft-2026-03-04-a'. Not the service build SHA — you need to "
        "be able to say which trained model produced a given row months later."
    )
    text_sha256: str = Field(
        description="Hash of the input, never the input itself. Support messages "
        "and survey free text are other people's data."
    )
    latency_ms: Optional[float] = Field(
        default=None, description="Server-side inference time, filled in by the API"
    )
    prediction_id: Optional[int] = Field(
        default=None,
        description="Row id in the Supabase predictions table, if logging succeeded",
    )


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, description="Raw input text from the user")
    model_kind: ModelKind = Field(
        default="transformer",
        description="Which of the two served models should answer. The UI lets "
        "the user switch so the comparison is live, not just a table in a report.",
    )


class PredictBatchRequest(BaseModel):
    texts: List[str] = Field(
        min_length=1,
        max_length=64,
        description="Up to 64 texts. The cap is not arbitrary: a transformer "
        "forward pass holds activations for the whole batch in memory, and the "
        "free Render plan will be killed by the OOM reaper long before you run "
        "out of patience. Chunk larger jobs client-side.",
    )
    model_kind: ModelKind = "transformer"


class PredictBatchResponse(BaseModel):
    predictions: List[PredictionResult] = Field(
        description="Same length as the request, in the SAME ORDER. Callers zip "
        "this against their input; reordering silently mislabels everything."
    )
    count: int


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
class Metrics(BaseModel):
    """Held-out performance for one model. All four numbers, always.

    Reporting accuracy alone on an imbalanced corpus is the single most common
    way a text-classification write-up ends up wrong, so the schema does not let
    you report it alone.
    """

    accuracy: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0, description="For positive_label")
    recall: float = Field(ge=0.0, le=1.0, description="For positive_label")
    f1: float = Field(ge=0.0, le=1.0, description="Harmonic mean of the two above")
    positive_label: str = Field(
        description="The class the three class-specific numbers refer to"
    )
    n_examples: int = Field(ge=0, description="Size of the held-out split")
    support_positive: int = Field(
        default=0,
        ge=0,
        description="How many held-out rows actually carry positive_label. If "
        "this is 7, your recall has a resolution of about 14 percentage points "
        "and you should say so rather than quoting three decimal places.",
    )


# ---------------------------------------------------------------------------
# GET /runs — one row per TRAINING run
# ---------------------------------------------------------------------------
class Run(BaseModel):
    """One training run: what you trained, how, and what it scored.

    Note what this is NOT: it is not a record of a served prediction. Those live
    in a separate table with a separate endpoint, because one training run
    answers thousands of requests and you need both halves to reconstruct what
    happened.
    """

    id: int
    model_kind: ModelKind
    dataset_name: Optional[str] = None
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Everything needed to re-run the training: vectorizer "
        "settings and C for the baseline; checkpoint, learning rate, epochs, "
        "batch size, max length and seed for the transformer. A run you cannot "
        "reproduce is an anecdote.",
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="The Metrics model, as JSON. Kept loose here so you can add "
        "your own numbers without a migration.",
    )
    model_version: str
    n_train: Optional[int] = None
    n_eval: Optional[int] = None
    created_at: Optional[str] = None


class RunsResponse(BaseModel):
    runs: List[Run]
