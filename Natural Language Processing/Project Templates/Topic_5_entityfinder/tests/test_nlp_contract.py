"""YOUR TO-DO LIST, WRITTEN AS TESTS.

Every test here fails on a fresh fork with NotImplementedError. Each one
describes a behaviour that ``api/nlp.py`` must have. Work top to bottom:

    pytest -m contract -x        # stop at the first thing left to do

These are deliberately loose about *how* — they check the contract, not your
algorithm. Passing them is necessary, not sufficient: the report, the error
analysis, and the review queue are where you show you understood what you built.

Two sections deserve more of your attention than the rest. The decoding tests
catch the bug that silently costs you recall, and the scoring tests catch the
bug that silently inflates your numbers. Neither needs a model, a GPU, or a
network connection, so there is no excuse for leaving them until the last night.
"""
from __future__ import annotations

import pytest

from api import nlp
from shared.schemas import TokenPrediction

pytestmark = pytest.mark.contract


def _tp(token, tag, start, end, conf=0.9):
    return TokenPrediction(
        token=token, tag=tag, start_char=start, end_char=end, confidence=conf
    )


# ---------------------------------------------------------------------------
# 1. The dataset. Marked network because it downloads a corpus.
# ---------------------------------------------------------------------------
@pytest.mark.network
def test_profile_dataset_counts_instead_of_guessing():
    profile = nlp.profile_dataset()
    assert profile.name
    assert profile.splits, "report a sentence count per split"
    assert all(v > 0 for v in profile.splits.values())
    assert len(profile.entity_types) >= 4
    assert profile.label_distribution, "report the token count for every BIO tag"
    # The O tag has to be in there. If your distribution excludes it, you have
    # hidden the class imbalance that explains why token accuracy is misleading.
    assert "O" in profile.label_distribution
    assert profile.label_distribution["O"] > 0


# ---------------------------------------------------------------------------
# 2. CRF features
# ---------------------------------------------------------------------------
def test_word_features_includes_prefixes_suffixes_and_pos():
    tokens = ["Ada", "Lovelace", "worked", "in", "London"]
    pos = ["NNP", "NNP", "VBD", "IN", "NNP"]
    feats = nlp.word_features(tokens, 1, pos)
    assert isinstance(feats, dict) and feats

    keys = " ".join(feats.keys()).lower()
    values = [v for v in feats.values() if isinstance(v, str)]
    assert "suffix" in keys or any(v == "ace" for v in values), "suffix feature required"
    assert "prefix" in keys or any(v == "Lov" for v in values), "prefix feature required"
    assert "pos" in keys, "a POS feature is required by the assignment"


def test_word_features_can_see_the_neighbouring_words():
    # "Washington" is a person, a place, or an organisation depending entirely
    # on its neighbours. A CRF that cannot see them cannot get this right.
    tokens = ["George", "Washington", "Bridge"]
    feats = nlp.word_features(tokens, 1, ["NNP", "NNP", "NNP"])
    flat = " ".join(f"{k}={v}" for k, v in feats.items()).lower()
    assert "george" in flat, "include features from the previous token"
    assert "bridge" in flat, "include features from the next token"


def test_word_features_marks_the_sentence_edges():
    tokens = ["London", "calling"]
    first = nlp.word_features(tokens, 0, ["NNP", "VBG"])
    last = nlp.word_features(tokens, 1, ["NNP", "VBG"])
    flat_first = " ".join(str(k) for k in first.keys()).upper()
    flat_last = " ".join(str(k) for k in last.keys()).upper()
    assert "BOS" in flat_first
    assert "EOS" in flat_last


def test_sentence_features_is_one_dict_per_token():
    tokens = ["Ada", "worked", "in", "London"]
    feats = nlp.sentence_features(tokens, ["NNP", "VBD", "IN", "NNP"])
    assert len(feats) == len(tokens)
    assert all(isinstance(f, dict) for f in feats)


# ---------------------------------------------------------------------------
# 3. BIO decoding — no model required, and the highest bug-per-line density in
#    the whole assignment. Write this first.
# ---------------------------------------------------------------------------
def test_decode_bio_spans_finds_a_multi_token_entity():
    text = "Ada Lovelace worked in London."
    preds = [
        _tp("Ada", "B-PER", 0, 3),
        _tp("Lovelace", "I-PER", 4, 12),
        _tp("worked", "O", 13, 19),
        _tp("in", "O", 20, 22),
        _tp("London", "B-LOC", 23, 29),
        _tp(".", "O", 29, 30),
    ]
    ents = nlp.decode_bio_spans(preds, text)
    assert len(ents) == 2
    assert (ents[0].start_char, ents[0].end_char, ents[0].entity_type) == (0, 12, "PER")
    assert (ents[1].start_char, ents[1].end_char, ents[1].entity_type) == (23, 29, "LOC")


def test_decoded_entity_text_is_sliced_from_the_original():
    # Two spaces between the words. If you rebuild the surface by joining tokens
    # you get "New York"; the reviewer's document says "New  York" and the
    # offsets they are shown no longer line up with what they read.
    text = "New  York is big."
    preds = [
        _tp("New", "B-LOC", 0, 3),
        _tp("York", "I-LOC", 5, 9),
        _tp("is", "O", 10, 12),
    ]
    ents = nlp.decode_bio_spans(preds, text)
    assert len(ents) == 1
    assert ents[0].text == text[ents[0].start_char : ents[0].end_char]
    assert ents[0].text == "New  York"


def test_adjacent_entities_of_the_same_type_stay_separate():
    # This is the entire reason the B- prefix exists. Merge these two and your
    # model reports one location that does not exist and misses two that do.
    text = "Paris London"
    preds = [_tp("Paris", "B-LOC", 0, 5), _tp("London", "B-LOC", 6, 12)]
    ents = nlp.decode_bio_spans(preds, text)
    assert len(ents) == 2


def test_decode_handles_a_stray_continuation_tag():
    # An I- tag with nothing open. Real models emit this all the time. Whatever
    # policy you chose, the function must not crash and must not invent a span
    # that runs off the front of the sentence.
    text = "Lovelace worked here"
    preds = [
        _tp("Lovelace", "I-PER", 0, 8),
        _tp("worked", "O", 9, 15),
        _tp("here", "O", 16, 20),
    ]
    ents = nlp.decode_bio_spans(preds, text)
    assert len(ents) <= 1
    for e in ents:
        assert e.start_char >= 0
        assert e.text == text[e.start_char : e.end_char]


def test_decode_gives_every_entity_a_usable_confidence():
    text = "Ada Lovelace"
    preds = [_tp("Ada", "B-PER", 0, 3, 0.99), _tp("Lovelace", "I-PER", 4, 12, 0.31)]
    ents = nlp.decode_bio_spans(preds, text)
    assert len(ents) == 1
    # Min, mean, or something you can defend — but not a constant, and not the
    # score of whichever token happened to be last.
    assert 0.0 <= ents[0].confidence <= 1.0
    assert ents[0].confidence <= 0.99


def test_decode_of_an_all_O_sentence_is_empty():
    text = "nothing to see here"
    preds = [_tp("nothing", "O", 0, 7), _tp("to", "O", 8, 10)]
    assert nlp.decode_bio_spans(preds, text) == []


# ---------------------------------------------------------------------------
# 4. Extraction end to end. Marked network: it loads a model.
# ---------------------------------------------------------------------------
@pytest.mark.network
def test_extract_entities_returns_spans_that_index_the_input():
    text = "Ada Lovelace worked in London for the Analytical Society."
    ents = nlp.extract_entities(text, "transformer")
    assert ents, "a sentence with three obvious entities should not come back empty"
    for e in ents:
        assert text[e.start_char : e.end_char] == e.text
        assert 0.0 <= e.confidence <= 1.0
        assert e.entity_type


@pytest.mark.network
def test_both_models_answer_with_the_same_shape():
    # The comparison tab depends on this. Same type in, same type out, so the UI
    # never needs to know which model produced a row.
    text = "London is in England."
    for model in ("transformer", "crf"):
        ents = nlp.extract_entities(text, model)
        assert isinstance(ents, list)
        for e in ents:
            assert text[e.start_char : e.end_char] == e.text


def test_empty_text_extracts_nothing_rather_than_raising():
    assert nlp.extract_entities("   ", "transformer") == []


def test_unknown_model_name_raises_value_error():
    with pytest.raises(ValueError):
        nlp.load_model("definitely-not-a-real-model-id-42")


# ---------------------------------------------------------------------------
# 5. Entity-level scoring. This is the number your report lives or dies on.
# ---------------------------------------------------------------------------
def test_exact_match_scores_one():
    gold = [(0, 12, "PER"), (23, 29, "LOC")]
    scores = nlp.entity_level_scores(gold, list(gold))
    assert scores.precision == pytest.approx(1.0)
    assert scores.recall == pytest.approx(1.0)
    assert scores.f1 == pytest.approx(1.0)
    assert scores.support == 2


def test_right_type_wrong_boundary_is_a_full_miss():
    # Gold "New York Stock Exchange", predicted "York Stock Exchange": the type
    # is right and the answer is useless. One false positive AND one false
    # negative — strictly worse than predicting nothing.
    gold = [(0, 23, "ORG")]
    predicted = [(4, 23, "ORG")]
    scores = nlp.entity_level_scores(gold, predicted)
    assert scores.precision == pytest.approx(0.0)
    assert scores.recall == pytest.approx(0.0)
    assert scores.false_positives == 1
    assert scores.false_negatives == 1


def test_right_boundary_wrong_type_is_also_a_miss():
    gold = [(0, 12, "PER")]
    predicted = [(0, 12, "ORG")]
    scores = nlp.entity_level_scores(gold, predicted)
    assert scores.f1 == pytest.approx(0.0)


def test_partial_credit_is_computed_the_ordinary_way():
    gold = [(0, 3, "PER"), (10, 16, "LOC"), (20, 24, "ORG")]
    predicted = [(0, 3, "PER"), (10, 16, "LOC"), (30, 34, "ORG")]
    scores = nlp.entity_level_scores(gold, predicted)
    assert scores.precision == pytest.approx(2 / 3)
    assert scores.recall == pytest.approx(2 / 3)
    assert scores.f1 == pytest.approx(2 / 3)


def test_predicting_nothing_is_recall_zero_not_a_crash():
    scores = nlp.entity_level_scores([(0, 3, "PER")], [])
    assert scores.recall == pytest.approx(0.0)
    assert scores.support == 1


def test_empty_everything_is_zeros_not_a_zero_division():
    scores = nlp.entity_level_scores([], [])
    assert scores.support == 0
    assert scores.f1 == pytest.approx(0.0)


def test_per_type_breakdown_is_populated():
    # The aggregate is dominated by whichever type is most common. The type you
    # are worst at is the one the report should be about, so report both.
    gold = [(0, 3, "PER"), (10, 16, "LOC")]
    predicted = [(0, 3, "PER"), (10, 16, "ORG")]
    scores = nlp.entity_level_scores(gold, predicted)
    assert "PER" in scores.per_type
    assert scores.per_type["PER"]["f1"] == pytest.approx(1.0)


def test_a_duplicate_prediction_cannot_match_the_same_gold_twice():
    gold = [(0, 3, "PER")]
    predicted = [(0, 3, "PER"), (0, 3, "PER")]
    scores = nlp.entity_level_scores(gold, predicted)
    assert scores.true_positives == 1
    assert scores.false_positives == 1
