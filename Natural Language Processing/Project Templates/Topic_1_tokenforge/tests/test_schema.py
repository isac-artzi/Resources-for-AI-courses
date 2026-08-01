"""Schema tests — the JSON contract between the two clouds.

These also pass on a fresh fork. They exist so that if you change
shared/schemas.py in a way that breaks the UI, you find out here rather than in
a deployed app.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.schemas import PreprocessRequest, TokenizeRequest, TokenizerResult


def test_preprocess_defaults_are_the_documented_ones():
    req = PreprocessRequest(text="hello world")
    assert req.options.lowercase is True
    assert req.options.lemmatize is True
    assert req.options.stem is False


def test_empty_text_is_rejected_before_it_reaches_your_code():
    with pytest.raises(ValidationError):
        PreprocessRequest(text="")


def test_tokenize_requires_at_least_two_tokenizers():
    with pytest.raises(ValidationError):
        TokenizeRequest(text="hello", tokenizers=["gpt2"])


def test_tokenizer_result_round_trips():
    r = TokenizerResult(
        tokenizer_name="gpt2",
        algorithm="BPE",
        tokens=["hello", "Ġworld"],
        token_ids=[1, 2],
        token_count=2,
        unknown_count=0,
        oov_rate=0.0,
        vocab_size=50257,
    )
    assert TokenizerResult(**r.model_dump()) == r
