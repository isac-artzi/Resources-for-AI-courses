"""Schema tests — the JSON contract between the two clouds.

These also pass on a fresh fork. They exist so that if you change
shared/schemas.py in a way that breaks the UI, you find out here rather than in
a deployed app.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.schemas import (
    DecodingParams,
    GenerateRequest,
    GenerateResponse,
    Generation,
    RateRequest,
    TrainingConfig,
)


def test_decoding_defaults_are_the_documented_ones():
    params = GenerateRequest(prompt="hello").params
    assert params.strategy == "top_p"
    assert params.top_p == 0.95
    assert params.num_beams == 4
    assert params.seed is None


def test_empty_prompt_is_rejected_before_it_reaches_your_code():
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="")


def test_unknown_strategy_is_rejected():
    """The five strategies are a closed set. A typo must fail loudly here, not
    silently fall through to greedy decoding in your generate()."""
    with pytest.raises(ValidationError):
        DecodingParams(strategy="nucleus")


def test_decoding_bounds_are_enforced():
    with pytest.raises(ValidationError):
        DecodingParams(top_p=1.5)
    with pytest.raises(ValidationError):
        DecodingParams(max_new_tokens=0)
    with pytest.raises(ValidationError):
        DecodingParams(temperature=-0.5)


def test_rating_must_be_one_to_five():
    with pytest.raises(ValidationError):
        RateRequest(generation_id=1, rater_id="rater-a", rating=6)
    with pytest.raises(ValidationError):
        RateRequest(generation_id=1, rater_id="", rating=3)


def test_generate_response_round_trips():
    r = GenerateResponse(
        prompt_sha256="0" * 64,
        generated_text="the harbour was quiet",
        strategy="top_p",
        params=DecodingParams(),
        prompt_token_count=4,
        generated_token_count=4,
        distinct_1=1.0,
        distinct_2=1.0,
    )
    assert GenerateResponse(**r.model_dump()) == r


def test_no_schema_carries_a_raw_prompt_field():
    """The privacy promise, enforced by a test.

    Nothing that gets persisted may have a field holding the user's prompt. If
    you add one to make the History tab prettier, this fails, and it is meant to.
    """
    for model in (GenerateResponse, Generation):
        fields = set(model.model_fields)
        assert "prompt" not in fields
        assert "prompt_text" not in fields
    assert "prompt_sha256" in Generation.model_fields


def test_training_config_records_what_the_report_has_to_state():
    cfg = TrainingConfig(base_model="gpt2", model_version="gentext-v1")
    dumped = cfg.model_dump()
    for key in ("base_model", "model_version", "epochs", "learning_rate", "method"):
        assert key in dumped
