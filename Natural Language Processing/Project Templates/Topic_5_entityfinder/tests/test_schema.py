"""Schema tests — the JSON contract between the two clouds.

These also pass on a fresh fork. They exist so that if you change
shared/schemas.py in a way that breaks the UI or the review write path, you find
out here rather than in a deployed app with a reviewer stuck.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.schemas import Entity, ExtractRequest, ReviewRequest, TokenPrediction


def test_extract_defaults_to_the_transformer():
    req = ExtractRequest(text="Ada Lovelace worked in London.")
    assert req.model == "transformer"


def test_empty_text_is_rejected_before_it_reaches_your_code():
    with pytest.raises(ValidationError):
        ExtractRequest(text="")


def test_an_unknown_model_name_is_rejected_at_the_edge():
    with pytest.raises(ValidationError):
        ExtractRequest(text="hello", model="lstm")


def test_entity_round_trips():
    e = Entity(
        text="Ada Lovelace",
        start_char=0,
        end_char=12,
        entity_type="PER",
        confidence=0.97,
    )
    assert Entity(**e.model_dump()) == e


def test_confidence_outside_zero_to_one_is_rejected():
    # A raw logit is not a confidence. If this fires, you are passing the score
    # through without a softmax somewhere.
    with pytest.raises(ValidationError):
        Entity(
            text="Ada", start_char=0, end_char=3, entity_type="PER", confidence=4.2
        )


def test_a_span_must_have_positive_length():
    with pytest.raises(ValidationError):
        Entity(text="", start_char=5, end_char=0, entity_type="PER", confidence=0.5)


def test_token_prediction_carries_offsets():
    tp = TokenPrediction(token="Ada", tag="B-PER", start_char=0, end_char=3, confidence=0.9)
    assert tp.end_char > tp.start_char


def test_accept_needs_no_correction_fields():
    req = ReviewRequest(entity_id=1, decision="accept", reviewer_id="r1")
    assert req.corrected_type is None


def test_a_correction_that_corrects_nothing_is_rejected():
    # "correct" with no corrected_* fields is a reviewer clicking the wrong
    # button. Catching it here beats storing a row that says a fix happened and
    # records no fix.
    with pytest.raises(ValidationError):
        ReviewRequest(entity_id=1, decision="correct", reviewer_id="r1")


def test_a_corrected_span_must_be_ordered():
    with pytest.raises(ValidationError):
        ReviewRequest(
            entity_id=1,
            decision="correct",
            reviewer_id="r1",
            corrected_start_char=10,
            corrected_end_char=4,
        )


def test_a_review_needs_a_reviewer():
    with pytest.raises(ValidationError):
        ReviewRequest(entity_id=1, decision="accept", reviewer_id="")
