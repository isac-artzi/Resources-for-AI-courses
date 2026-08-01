"""Schema tests — the JSON contract between the two clouds.

These also pass on a fresh fork. They exist so that if you change
shared/schemas.py in a way that breaks the UI, you find out here rather than in
a deployed app.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.schemas import ConfusionMatrix, Run, TaggedToken, TagRequest


def test_tag_request_defaults_to_the_baseline():
    req = TagRequest(sentence="They book the flight.")
    assert req.model == "baseline"


def test_empty_sentence_is_rejected_before_it_reaches_your_code():
    with pytest.raises(ValidationError):
        TagRequest(sentence="")


def test_only_the_two_documented_models_are_accepted():
    with pytest.raises(ValidationError):
        TagRequest(sentence="hello", model="lookup-v2")


def test_tagged_token_round_trips():
    t = TaggedToken(token="book", tag="VERB", confidence=0.91, used_fallback=False)
    assert TaggedToken(**t.model_dump()) == t


def test_a_lookup_token_may_carry_no_confidence():
    """The baseline has no probability. None is the honest value, not 1.0."""
    t = TaggedToken(token="Zurich", tag="PROPN", used_fallback=True)
    assert t.confidence is None
    assert t.used_fallback is True


def test_run_carries_hyperparameters_for_both_models():
    """The lookup table has hyperparameters too — they are decisions, not numbers."""
    run = Run(
        id=1,
        model="baseline",
        tagset="UPOS",
        hyperparameters={"lowercase_keys": False, "tie_break": "corpus frequency"},
        accuracy=0.9,
        macro_f1=0.7,
        metrics={"confusion": {"labels": ["NOUN", "VERB"], "matrix": [[8, 2], [1, 9]]}},
        model_version="tagwise-v1",
    )
    assert Run(**run.model_dump()) == run
    assert run.hyperparameters["tie_break"]


def test_confusion_matrix_is_square_over_its_own_labels():
    cm = ConfusionMatrix(labels=["NOUN", "VERB"], matrix=[[8, 2], [1, 9]])
    assert len(cm.matrix) == len(cm.labels)
    assert all(len(row) == len(cm.labels) for row in cm.matrix)
