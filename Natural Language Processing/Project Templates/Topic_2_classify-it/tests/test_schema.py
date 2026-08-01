"""Schema tests — the JSON contract between the two clouds.

These also pass on a fresh fork. They exist so that if you change
shared/schemas.py in a way that breaks the UI, you find out here rather than in
a deployed app.

Most of this file is about ONE rule: single-label, exactly two label values.
That rule is enforced in the type system and in a validator rather than in a
comment, because "we meant to only use binary datasets" is not something a
grader can check and not something a future maintainer will notice.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.schemas import (
    LabeledExample,
    LabelSchema,
    Metrics,
    PredictBatchRequest,
    PredictionResult,
    PredictRequest,
)


def _schema(**overrides) -> dict:
    base = dict(
        labels=["urgent", "routine"],
        positive_label="urgent",
        class_counts={"urgent": 800, "routine": 4200},
        dataset_name="demo",
        n_rows=5000,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Request defaults
# ---------------------------------------------------------------------------
def test_predict_defaults_to_the_transformer():
    req = PredictRequest(text="hello world")
    assert req.model_kind == "transformer"


def test_empty_text_is_rejected_before_it_reaches_your_code():
    with pytest.raises(ValidationError):
        PredictRequest(text="")


def test_batch_is_capped_so_the_free_plan_survives():
    with pytest.raises(ValidationError):
        PredictBatchRequest(texts=["x"] * 65)


def test_batch_of_one_is_fine():
    assert len(PredictBatchRequest(texts=["x"]).texts) == 1


# ---------------------------------------------------------------------------
# The single-label rule
# ---------------------------------------------------------------------------
def test_a_labeled_example_carries_exactly_one_label():
    ex = LabeledExample(text="my order never arrived", label="urgent")
    assert ex.label == "urgent"
    # A list of labels is multilabel, which is out of scope for this product.
    with pytest.raises(ValidationError):
        LabeledExample(text="my order never arrived", label=["urgent", "billing"])


def test_two_labels_is_the_only_accepted_size():
    ok = LabelSchema(**_schema())
    assert ok.labels == ["urgent", "routine"]

    # Three labels is multiclass — a different loss reduction, a different
    # definition of precision, and a later course.
    with pytest.raises(ValidationError):
        LabelSchema(
            **_schema(
                labels=["urgent", "routine", "spam"],
                class_counts={"urgent": 1, "routine": 1, "spam": 1},
            )
        )

    # One label is not a classification problem at all.
    with pytest.raises(ValidationError):
        LabelSchema(**_schema(labels=["urgent"], class_counts={"urgent": 1}))


def test_the_two_labels_must_be_distinct():
    with pytest.raises(ValidationError):
        LabelSchema(**_schema(labels=["urgent", "urgent"], class_counts={"urgent": 1}))


def test_positive_label_must_be_one_of_the_labels():
    with pytest.raises(ValidationError):
        LabelSchema(**_schema(positive_label="spam"))


def test_class_counts_must_agree_with_the_label_set():
    with pytest.raises(ValidationError):
        LabelSchema(**_schema(class_counts={"urgent": 10, "spam": 10}))


def test_minority_share_is_the_number_that_makes_accuracy_suspicious():
    s = LabelSchema(**_schema(class_counts={"urgent": 250, "routine": 4750}))
    assert s.minority_share() == pytest.approx(0.05)
    # With a 5% minority class, always answering "routine" scores 95% accuracy.


def test_minority_share_is_none_when_counts_are_unknown():
    assert LabelSchema(**_schema(class_counts={})).minority_share() is None


# ---------------------------------------------------------------------------
# Serving and metric shapes
# ---------------------------------------------------------------------------
def test_prediction_result_round_trips():
    r = PredictionResult(
        label="urgent",
        probability=0.87,
        model_kind="transformer",
        model_version="distilbert-ft-v2",
        text_sha256="0" * 64,
    )
    assert PredictionResult(**r.model_dump()) == r


def test_probability_outside_zero_to_one_is_rejected():
    for bad in (-0.01, 1.5):
        with pytest.raises(ValidationError):
            PredictionResult(
                label="urgent",
                probability=bad,
                model_kind="baseline",
                model_version="v1",
                text_sha256="0" * 64,
            )


def test_metrics_requires_all_four_numbers_and_a_positive_label():
    # Accuracy alone is not a report. The schema will not let you file one.
    with pytest.raises(ValidationError):
        Metrics(accuracy=0.95, positive_label="urgent", n_examples=1000)

    m = Metrics(
        accuracy=0.95,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        positive_label="urgent",
        n_examples=1000,
        support_positive=50,
    )
    # This object is internally consistent and describes a model that never once
    # predicted the positive class. That is exactly why F1 is not optional.
    assert m.accuracy > m.f1
