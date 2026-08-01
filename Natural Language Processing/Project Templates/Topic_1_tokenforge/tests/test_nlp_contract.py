"""YOUR TO-DO LIST, WRITTEN AS TESTS.

Every test here fails on a fresh fork with NotImplementedError. Each one
describes a behaviour that ``api/nlp.py`` must have. Work top to bottom:

    pytest -m contract -x        # stop at the first thing left to do

These are deliberately loose about *how* — they check the contract, not your
algorithm. Passing them is necessary, not sufficient: the report and the Compare
tab are where you show you understood what you built.
"""
from __future__ import annotations

import pytest

from api import nlp
from shared.schemas import PreprocessOptions

pytestmark = pytest.mark.contract


# ---------------------------------------------------------------------------
# 1. word_tokenize
# ---------------------------------------------------------------------------
def test_word_tokenize_returns_tokens_in_order():
    tokens = nlp.word_tokenize("The quick brown fox")
    assert tokens == ["The", "quick", "brown", "fox"]


def test_word_tokenize_handles_empty_input():
    assert nlp.word_tokenize("") == []


def test_word_tokenize_keeps_contractions_whole():
    # "don't" is ONE word. Splitting it into "don" + "t" is the classic failure
    # and it changes the meaning of every negated sentence in your corpus.
    assert "don't" in [t.lower() for t in nlp.word_tokenize("I don't think so")]


# ---------------------------------------------------------------------------
# 2. The classical steps
# ---------------------------------------------------------------------------
def test_strip_punctuation_drops_punct_only_tokens():
    out = nlp.strip_punctuation(["hello", ",", "world", "!!!"])
    assert "," not in out and "!!!" not in out
    assert "hello" in out and "world" in out


def test_strip_punctuation_does_not_mutate_its_input():
    original = ["hello", ","]
    nlp.strip_punctuation(original)
    assert original == ["hello", ","]


def test_remove_stopwords_is_case_insensitive_but_preserves_case():
    out = nlp.remove_stopwords(["The", "Signal", "and", "Noise"])
    assert "The" not in out and "and" not in out
    assert out == ["Signal", "Noise"]  # case of survivors is untouched


def test_stem_is_crude_and_that_is_expected():
    assert nlp.stem(["running"]) == ["run"]
    # Short tokens are left alone — the rules do more harm than good there.
    assert nlp.stem(["cat"]) == ["cat"]


def test_lemmatize_returns_real_words():
    out = nlp.lemmatize(["was", "mice"])
    assert out == ["be", "mouse"]


# ---------------------------------------------------------------------------
# 3. preprocess
# ---------------------------------------------------------------------------
def test_preprocess_records_every_step_it_ran():
    result = nlp.preprocess(
        "The cats were running quickly!",
        PreprocessOptions(lowercase=True, strip_punctuation=True, remove_stopwords=True),
    )
    assert result.steps_applied, "steps_applied must not be empty — it is the audit trail"
    assert len(result.steps_applied) >= 3
    assert result.token_count_before >= result.token_count_after


def test_preprocess_with_everything_off_changes_nothing():
    opts = PreprocessOptions(
        lowercase=False,
        strip_punctuation=False,
        remove_stopwords=False,
        remove_digits=False,
        stem=False,
        lemmatize=False,
    )
    result = nlp.preprocess("The cats ran.", opts)
    assert result.cleaned_tokens == result.original_tokens
    assert result.token_count_before == result.token_count_after


def test_preprocess_prefers_lemmatize_when_both_are_requested():
    result = nlp.preprocess("mice", PreprocessOptions(stem=True, lemmatize=True))
    assert any("lemmat" in s.lower() for s in result.steps_applied)
    assert not any(s.lower().startswith("stem") for s in result.steps_applied)


# ---------------------------------------------------------------------------
# 4. Subword tokenization
#    These download a tokenizer the first time they run. If you are offline,
#    deselect them with: pytest -m "contract and not network"
# ---------------------------------------------------------------------------
@pytest.mark.network
def test_subword_tokenize_fills_every_field():
    r = nlp.subword_tokenize("tokenization of unhappiness", "bert-base-uncased")
    assert r.tokenizer_name == "bert-base-uncased"
    assert r.algorithm
    assert r.tokens and len(r.tokens) == r.token_count
    assert len(r.token_ids) == r.token_count
    assert r.vocab_size > 1000
    assert 0.0 <= r.oov_rate <= 1.0


@pytest.mark.network
def test_subword_tokenize_splits_a_rare_word_into_pieces():
    # A word no tokenizer has as a single vocabulary entry must come back as
    # several pieces. If it comes back as one, you are not calling the tokenizer.
    r = nlp.subword_tokenize("antidisestablishmentarianism", "bert-base-uncased")
    assert r.token_count > 1


def test_unknown_tokenizer_name_raises_value_error():
    with pytest.raises(ValueError):
        nlp.load_tokenizer("definitely-not-a-real-tokenizer-id-42")


# ---------------------------------------------------------------------------
# 5. vocabulary_overlap
# ---------------------------------------------------------------------------
def test_vocabulary_overlap_of_one_result_is_empty():
    from shared.schemas import TokenizerResult

    one = TokenizerResult(
        tokenizer_name="a",
        algorithm="BPE",
        tokens=["x"],
        token_ids=[1],
        token_count=1,
        unknown_count=0,
        oov_rate=0.0,
        vocab_size=10,
    )
    assert nlp.vocabulary_overlap([one]) == {}


def test_vocabulary_overlap_is_jaccard():
    from shared.schemas import TokenizerResult

    def mk(name, toks):
        return TokenizerResult(
            tokenizer_name=name,
            algorithm="BPE",
            tokens=toks,
            token_ids=list(range(len(toks))),
            token_count=len(toks),
            unknown_count=0,
            oov_rate=0.0,
            vocab_size=10,
        )

    out = nlp.vocabulary_overlap([mk("a", ["x", "y"]), mk("b", ["y", "z"])])
    # |{y}| / |{x,y,z}| = 1/3
    assert pytest.approx(1 / 3, abs=1e-6) == out["a|b"]
