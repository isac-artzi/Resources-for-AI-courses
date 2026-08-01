"""Schema tests — the JSON contract between the clouds.

These also pass on a fresh fork. They exist so that if you change
shared/schemas.py in a way that breaks the UI or the metrics payload, you find
out here rather than in a deployed app three minutes before your demo.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.schemas import (
    AspectSentiment,
    CalibrationBin,
    DocumentMetrics,
    PredictBatchRequest,
    PredictionRecord,
    PredictRequest,
    SentimentPrediction,
    SliceMetrics,
)


def test_predict_defaults_are_the_documented_ones():
    req = PredictRequest(text="the ending is perfect")
    assert req.include_aspects is True


def test_empty_text_is_rejected_before_it_reaches_your_code():
    with pytest.raises(ValidationError):
        PredictRequest(text="")


def test_batch_is_capped():
    # 64 is the documented ceiling: an unbounded batch on a CPU-only free plan
    # finds the gateway timeout instead of the model.
    PredictBatchRequest(texts=["a"] * 64)
    with pytest.raises(ValidationError):
        PredictBatchRequest(texts=["a"] * 65)
    with pytest.raises(ValidationError):
        PredictBatchRequest(texts=[])


def test_probabilities_must_be_probabilities():
    with pytest.raises(ValidationError):
        SentimentPrediction(
            label="positive",
            probability_positive=1.4,
            raw_probability_positive=0.9,
            confidence=0.9,
            calibrated=True,
        )


def test_document_label_set_is_closed():
    # If you add a "neutral" class you must widen DocumentLabel in
    # shared/schemas.py first — and then explain in the model card what data
    # taught the model a class it was never trained on.
    with pytest.raises(ValidationError):
        SentimentPrediction(
            label="neutral",
            probability_positive=0.5,
            raw_probability_positive=0.5,
            confidence=0.5,
            calibrated=False,
        )


def test_aspects_may_be_absent_from_a_review():
    a = AspectSentiment(aspect="service", label="not_mentioned", score=0.0)
    assert a.evidence == []


def test_sentiment_prediction_round_trips():
    p = SentimentPrediction(
        label="negative",
        probability_positive=0.12,
        raw_probability_positive=0.03,
        confidence=0.88,
        calibrated=True,
        model_name="transformer",
    )
    assert SentimentPrediction(**p.model_dump()) == p


def test_prediction_record_tolerates_the_columns_the_database_returns():
    # Supabase hands back every column; the record model must accept the row as
    # it comes out, or GET /audit breaks the first time you add a column.
    row = {
        "id": 1,
        "text_sha256": "0" * 64,
        "char_count": 42,
        "label": "positive",
        "probability_positive": 0.81,
        "confidence": 0.81,
        "calibrated": True,
        "aspects": [{"aspect": "acting", "label": "positive", "score": 0.7, "evidence": []}],
        "model_name": "transformer",
        "model_version": "moodlens-v2",
        "created_at": "2026-01-01T00:00:00Z",
    }
    assert PredictionRecord(**row).aspects[0]["aspect"] == "acting"


def test_metrics_shapes_are_what_the_ui_reads():
    """The Model Performance tab reads runs.metrics with these field names."""
    m = DocumentMetrics(
        model_name="tfidf-baseline",
        n=4,
        accuracy=0.75,
        macro_precision=0.75,
        macro_recall=0.75,
        macro_f1=0.75,
        confusion_matrix=[[2, 0], [1, 1]],
        calibration_bins=[
            CalibrationBin(
                bin_lower=0.8,
                bin_upper=0.9,
                mean_predicted=0.85,
                observed_positive_rate=0.5,
                count=4,
            )
        ],
    )
    assert m.labels == ["negative", "positive"]  # confusion matrix row/col order
    assert m.calibration_bins[0].mean_predicted != m.calibration_bins[0].observed_positive_rate


def test_slice_metrics_default_to_observed_and_can_be_marked_inferred():
    observed = SliceMetrics(
        slice_name="review_length", bucket="short", n=120, accuracy=0.7, macro_f1=0.69
    )
    assert observed.observed is True
    inferred = SliceMetrics(
        slice_name="genre",
        bucket="horror",
        n=40,
        accuracy=0.6,
        macro_f1=0.58,
        observed=False,
    )
    # The UI puts a warning on this row. An audit that slices on an attribute
    # your own model guessed is auditing the guess as much as the classifier.
    assert inferred.observed is False
