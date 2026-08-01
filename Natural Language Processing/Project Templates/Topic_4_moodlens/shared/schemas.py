"""Pydantic request/response models — the contract between the three clouds.

WHY THIS FILE EXISTS
--------------------
The UI (Streamlit Cloud) and the API (Render.com) are separate deployments that
only ever talk over HTTPS + JSON. This module is the single written-down
description of that JSON. Both tiers import it, so a change here is a change on
both sides at once and you cannot drift.

Two of the model families below are never returned by an endpoint:
``DocumentMetrics``, ``AspectMetrics`` and ``SliceMetrics`` describe what your
training script writes into the ``metrics`` column of the ``runs`` table. The
Model Performance and Bias Audit tabs read that column straight out of Supabase
and expect these shapes. Treat them as the contract for evaluation the same way
``PredictResponse`` is the contract for serving.

WHAT YOU SHOULD DO WITH IT
--------------------------
Read it first, before you write any NLP code. Every stub in ``api/nlp.py`` is
typed in terms of these models, so the shapes below tell you exactly what your
functions have to produce.

You MAY add fields. If your aspect extractor also returns a character span, add
it to ``AspectSentiment`` and render it in the UI. Do NOT rename or delete the
existing fields — the tests and the grading rubric reference them by name.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class SchemaModel(BaseModel):
    """Base class for every model in this file.

    It exists for one reason: pydantic reserves the ``model_`` prefix for its own
    methods and warns about any field that starts with it. Half the fields in a
    machine-learning API legitimately start with ``model_`` — ``model_version``,
    ``model_name`` — and renaming them to dodge a warning would make the JSON
    worse to read. So the namespace is opened here, once, deliberately.
    """

    model_config = ConfigDict(protected_namespaces=())


# ---------------------------------------------------------------------------
# Label sets.
#
# The document label set is binary because the corpus in the Build Steps (IMDb)
# is binary. That is a decision, not a law of nature: real review streams are
# full of mixed and neutral text, and a binary model has nowhere to put it, so
# it puts it in whichever class is closer. If you add "neutral", change it here
# first, then say in the MODEL_CARD what training data taught the model the new
# class — a label you never trained on is a label you cannot predict.
#
# Aspects get a fourth value, "not_mentioned". A review that never talks about
# the acting has no acting sentiment, and forcing one is the single easiest way
# to make an aspect breakdown that looks informative and is not.
# ---------------------------------------------------------------------------
DocumentLabel = Literal["negative", "positive"]
AspectLabel = Literal["negative", "neutral", "positive", "not_mentioned"]


# ---------------------------------------------------------------------------
# Infrastructure responses (already implemented — you should not need to touch)
# ---------------------------------------------------------------------------
class Health(SchemaModel):
    """GET /healthz — is the service up, and can it reach the database?"""

    status: str = Field(description='"ok" when the process is serving requests')
    database: str = Field(description='"ok", "unreachable", or "not_configured"')
    model_loaded: bool = Field(
        default=False,
        description="True once the classifier is in memory. On Render's free tier "
        "the first request after a cold start pays for the load; this flag is how "
        "you tell a slow service from a broken one.",
    )


class Version(SchemaModel):
    """GET /version — which build is live right now?

    Graders use this to confirm the URL in your README is the code you claim.
    """

    service: str
    git_sha: str = Field(description="Short commit SHA of the running build")
    model_version: str = Field(
        description="Free-text version of the served model, e.g. 'moodlens-v2'. "
        "Every logged prediction carries it, which is what lets you say later "
        "which build produced a bad answer."
    )
    base_model: str = Field(
        default="",
        description="Hugging Face id the classifier was fine-tuned from, e.g. "
        "'distilbert-base-uncased'. Part of the model card, so serve it.",
    )


# ---------------------------------------------------------------------------
# The two units of a MoodLens answer
# ---------------------------------------------------------------------------
class SentimentPrediction(SchemaModel):
    """Document-level sentiment for one text."""

    label: DocumentLabel
    probability_positive: float = Field(
        ge=0.0,
        le=1.0,
        description="P(positive) AFTER calibration. This is the number the UI "
        "shows and the number the calibration plot is scored on.",
    )
    raw_probability_positive: float = Field(
        ge=0.0,
        le=1.0,
        description="P(positive) straight out of the softmax, before calibration. "
        "Keep both: the gap between them is the only evidence that calibrating "
        "did anything.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Calibrated probability of the label that was actually "
        "predicted, i.e. max(p, 1 - p). Always >= 0.5 for a binary model.",
    )
    calibrated: bool = Field(
        description="False means no calibrator was fitted and the two "
        "probability fields are equal. Report it honestly; an uncalibrated "
        "probability is a score, not a probability."
    )
    model_name: str = Field(
        default="transformer",
        description='"transformer" or "tfidf-baseline" — which model answered. '
        "The baseline is not decoration; you have to report its numbers too.",
    )


class AspectSentiment(SchemaModel):
    """Sentiment attributed to one aspect of the thing being reviewed."""

    aspect: str = Field(description="One of the aspects you documented, e.g. 'acting'")
    label: AspectLabel
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in this aspect label. For 'not_mentioned' this is "
        "your confidence that the aspect is absent, not that it is positive.",
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="Snippets COPIED from the input text that support the label. "
        "They must be substrings of the input. An aspect breakdown with no "
        "evidence is an assertion; with evidence it is an argument, and it is "
        "the only part of this response a human reviewer can check quickly.",
    )


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------
class PredictRequest(SchemaModel):
    text: str = Field(min_length=1, description="One review, as the analyst pasted it")
    include_aspects: bool = Field(
        default=True,
        description="Aspect extraction is the expensive half of the request. "
        "Turning it off is how the batch endpoint stays usable.",
    )


class PredictResponse(SchemaModel):
    """Everything the Score Text and Aspect Breakdown tabs need."""

    text_sha256: str = Field(
        description="Hash of the input. The response carries it so the analyst "
        "can find this exact prediction in the audit log without the log ever "
        "holding the review text."
    )
    char_count: int
    sentiment: SentimentPrediction
    aspects: List[AspectSentiment] = Field(
        default_factory=list,
        description="One entry per aspect you defined — including the ones the "
        "review never mentions, labelled 'not_mentioned'. Silence is a finding.",
    )
    model_version: str
    prediction_id: Optional[int] = Field(
        default=None,
        description="Row id in the Supabase predictions table, if logging succeeded",
    )


# ---------------------------------------------------------------------------
# POST /predict_batch
# ---------------------------------------------------------------------------
class PredictBatchRequest(SchemaModel):
    texts: List[str] = Field(
        min_length=1,
        max_length=64,
        description="The cap is deliberate. A transformer on Render's free tier "
        "is CPU-only; an unbounded batch is how you discover the 30-second "
        "gateway timeout in front of an audience.",
    )
    include_aspects: bool = False


class PredictBatchResponse(SchemaModel):
    results: List[PredictResponse]
    count: int
    label_counts: Dict[str, int] = Field(
        default_factory=dict,
        description='How many of each label, e.g. {"positive": 7, "negative": 3}. '
        "The UI charts it; you get it for free from the results.",
    )


# ---------------------------------------------------------------------------
# GET /audit
# ---------------------------------------------------------------------------
class PredictionRecord(SchemaModel):
    """One served prediction, as stored. The audit view reads these."""

    id: int
    text_sha256: str = Field(
        description="Hash of the input, never the input itself. Reviews are "
        "someone's writing about someone else's work; you log what you need to "
        "reproduce a decision, not the text."
    )
    char_count: Optional[int] = None
    label: str
    probability_positive: Optional[float] = None
    confidence: Optional[float] = None
    calibrated: Optional[bool] = None
    aspects: List[dict] = Field(
        default_factory=list,
        description="The aspect breakdown exactly as served, as JSON. Stored with "
        "the prediction because an audit that cannot see which aspect drove the "
        "answer cannot explain the answer.",
    )
    model_name: Optional[str] = None
    model_version: str
    created_at: Optional[str] = None


class AuditResponse(SchemaModel):
    predictions: List[PredictionRecord]
    count: int


# ---------------------------------------------------------------------------
# Evaluation shapes — written into runs.metrics, read by the UI
# ---------------------------------------------------------------------------
class ClassMetrics(SchemaModel):
    """Precision, recall and F1 for one class."""

    precision: float
    recall: float
    f1: float
    support: int = Field(description="How many held-out items truly have this label")


class RocPoint(SchemaModel):
    """One point on the ROC curve: the model's behaviour at one threshold."""

    threshold: float
    fpr: float
    tpr: float


class CalibrationBin(SchemaModel):
    """One bin of the reliability diagram.

    ``mean_predicted`` is what the model claimed; ``observed_positive_rate`` is
    what actually happened. Plotting the first against itself gives a perfect
    diagonal and proves nothing — see the trap list in the README.
    """

    bin_lower: float
    bin_upper: float
    mean_predicted: float
    observed_positive_rate: float
    count: int


class DocumentMetrics(SchemaModel):
    """Held-out document-level metrics for ONE model.

    You produce two of these — one for the fine-tuned transformer and one for
    the TF-IDF baseline — and the Model Performance tab shows them side by side.
    A transformer that cannot beat TF-IDF on your split is a finding, and it is
    a finding you report rather than hide.
    """

    model_name: str
    n: int = Field(description="Size of the held-out split")
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float = Field(
        description="Macro, not micro. Averaged over classes so the small class "
        "counts as much as the large one."
    )
    per_class: Dict[str, ClassMetrics] = Field(default_factory=dict)
    labels: List[str] = Field(
        default_factory=lambda: ["negative", "positive"],
        description="Row/column order of confusion_matrix",
    )
    confusion_matrix: List[List[int]] = Field(
        default_factory=list,
        description="confusion_matrix[gold][predicted], indexed by `labels`",
    )
    roc_auc: Optional[float] = None
    roc_points: List[RocPoint] = Field(default_factory=list)
    calibration_bins: List[CalibrationBin] = Field(default_factory=list)


class AspectMetrics(SchemaModel):
    """Precision, recall and F1 for ONE aspect, over the aspect-labelled split."""

    aspect: str
    precision: float
    recall: float
    f1: float
    support: int
    n_evaluated: int = Field(
        default=0,
        description="How many documents carried a gold label for this aspect. If "
        "this is 40, say 40 — three-decimal F1 on forty examples is a number "
        "with an error bar you have not drawn.",
    )


class SliceMetrics(SchemaModel):
    """Performance on one bucket of one slice — the Bias Audit tab's raw material."""

    slice_name: str = Field(description='The attribute you sliced on, e.g. "review_length"')
    bucket: str = Field(description='The value of that attribute, e.g. "short"')
    n: int
    accuracy: float
    macro_f1: float
    observed: bool = Field(
        default=True,
        description="True if the attribute was OBSERVED in the data (a length you "
        "measured, a genre field the corpus ships). False if your own model or a "
        "heuristic inferred it. An inferred slice audits the inference as much as "
        "the classifier — see the README trap list — and the UI labels it as such.",
    )
