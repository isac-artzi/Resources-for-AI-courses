"""YOUR TO-DO LIST, WRITTEN AS TESTS.

Every test here fails on a fresh fork with NotImplementedError. Each one
describes a behaviour that ``api/nlp.py`` must have. Work top to bottom:

    pytest -m contract -x                       # stop at the first thing left to do
    pytest -m "contract and not network"        # skip the ones that load a model

These are deliberately loose about *how* — they check the contract, not your
algorithm. Several of them run on the six-sentence treebank in conftest.py,
which is small enough to check by eye and far too small to train on. Passing
them is necessary, not sufficient: the report, the confusion matrix, and the
worked examples of ambiguity are where you show you understood what you built.
"""
from __future__ import annotations

import pytest

from api import nlp
from shared.schemas import TaggedToken

pytestmark = pytest.mark.contract


# ---------------------------------------------------------------------------
# 1. The corpus
# ---------------------------------------------------------------------------
def test_load_treebank_returns_sentences_of_word_tag_pairs(tiny_conllu_file):
    sentences = nlp.load_treebank(tiny_conllu_file)
    assert len(sentences) == 2, "blank lines separate sentences; comments are not sentences"
    assert sentences[0][0] == ("The", "DET")
    assert sentences[0][-1] == (".", "PUNCT")


def test_load_treebank_skips_range_ids_and_empty_nodes(tiny_conllu_file):
    """Rows with an id like 8-9 or 5.1 are not words. Counting them corrupts
    every number you compute afterwards, and nothing crashes when you do."""
    words = [word for word, _ in nlp.load_treebank(tiny_conllu_file)[1]]
    assert words == ["They", "do", "n't", "book", "flights", "."]


def test_describe_corpus_reports_size_and_distribution(tiny_treebank):
    stats = nlp.describe_corpus(tiny_treebank)
    assert stats["n_sentences"] == 6
    assert stats["n_tokens"] == sum(len(s) for s in tiny_treebank)
    assert stats["tagset"] == sorted(set(tag for s in tiny_treebank for _, tag in s))
    assert pytest.approx(1.0, abs=1e-9) == sum(stats["tag_distribution"].values())


def test_describe_corpus_survives_an_empty_split():
    stats = nlp.describe_corpus([])
    assert stats["n_tokens"] == 0
    assert stats["tag_distribution"] == {}


# ---------------------------------------------------------------------------
# 2. The lookup baseline
# ---------------------------------------------------------------------------
def test_lookup_table_keeps_the_most_frequent_tag(tiny_treebank):
    table = nlp.build_lookup_table(tiny_treebank)
    # "book" is NOUN twice and VERB once; "flies" is NOUN twice and VERB once.
    assert table["book"] == "NOUN"
    assert table["flies"] == "NOUN"


def test_lookup_table_contains_only_words_it_has_seen(tiny_treebank):
    table = nlp.build_lookup_table(tiny_treebank)
    assert "aardvark" not in table
    assert all(tag in nlp.UNIVERSAL_POS_TAGS for tag in table.values())


def test_fallback_tag_always_returns_a_real_tag():
    for word in ["Zurich", "42", "quickly", "glorbish", "!!", "$"]:
        assert nlp.fallback_tag(word) in nlp.UNIVERSAL_POS_TAGS


def test_fallback_rules_do_the_obvious_things():
    assert nlp.fallback_tag("3.14") == "NUM"
    assert nlp.fallback_tag("Kowalczyk") == "PROPN"
    assert nlp.fallback_tag("brusquely") == "ADV"
    # An unknown word with no useful shape is a noun, because open classes grow
    # and closed classes do not.
    assert nlp.fallback_tag("glorbish") == "NOUN"


def test_tag_with_lookup_marks_the_words_it_had_to_guess(tiny_treebank):
    table = nlp.build_lookup_table(tiny_treebank)
    tagged = nlp.tag_with_lookup(["The", "book", "glorbished"], table)
    assert len(tagged) == 3
    assert all(isinstance(t, TaggedToken) for t in tagged)
    known, unknown = tagged[1], tagged[2]
    assert known.used_fallback is False
    assert unknown.used_fallback is True, "an unseen word must be flagged, not hidden"
    assert known.confidence is None, "the lookup has no probability; None, not 1.0"


def test_the_lookup_cannot_change_its_mind_about_an_ambiguous_word(tiny_treebank):
    """The finding the whole topic turns on, as an executable statement.

    'book' is a NOUN in 'read the book' and a VERB in 'book me a flight'. A tagger
    that is a function of the word alone gives it the same tag in both, so it must
    be wrong about one of them. Nothing you can do to the table fixes this; only
    looking at the neighbouring words does.
    """
    table = nlp.build_lookup_table(tiny_treebank)
    as_noun = nlp.tag_with_lookup(["read", "the", "book"], table)[-1].tag
    as_verb = nlp.tag_with_lookup(["They", "book", "flights"], table)[1].tag
    assert as_noun == as_verb


# ---------------------------------------------------------------------------
# 3. Aligning tags to subword pieces
#    Read the docstring in nlp.py before you make this pass. Getting it wrong is
#    silent: the model trains, the loss falls, and the accuracy is quietly wrong.
# ---------------------------------------------------------------------------
def test_alignment_labels_only_the_first_piece_of_each_word():
    tag_to_id = {"DET": 0, "NOUN": 1, "PUNCT": 2}
    # [CLS] the un ##happi ##ness . [SEP]  ->  words: "the", "unhappiness", "."
    word_ids = [None, 0, 1, 1, 1, 2, None]
    labels = nlp.align_tags_to_subwords(word_ids, ["DET", "NOUN", "PUNCT"], tag_to_id)
    assert labels == [-100, 0, 1, -100, -100, 2, -100]


def test_alignment_output_is_one_label_per_piece():
    tag_to_id = {"NOUN": 0, "VERB": 1}
    word_ids = [None, 0, 0, 0, 1, None]
    labels = nlp.align_tags_to_subwords(word_ids, ["NOUN", "VERB"], tag_to_id)
    assert len(labels) == len(word_ids)
    assert labels.count(-100) == 4  # two specials plus two continuation pieces


# ---------------------------------------------------------------------------
# 4. Serving one request
# ---------------------------------------------------------------------------
def test_split_words_separates_punctuation_from_words():
    tokens = nlp.split_words("They book the flight.")
    assert "flight" in tokens and "." in tokens
    assert "flight." not in tokens, "PUNCT cannot be predicted if it is glued to a noun"


def test_split_words_handles_empty_input():
    assert nlp.split_words("") == []


def test_tag_sentence_returns_one_tag_per_token():
    result = nlp.tag_sentence("They book the flight.", "baseline")
    assert result.model == "baseline"
    assert len(result.tag_sequence) == len(result.tokens)
    assert result.tag_sequence == [t.tag for t in result.tokens]
    assert result.unknown_count == sum(1 for t in result.tokens if t.used_fallback)


def test_tag_sentence_rejects_a_model_it_does_not_have():
    with pytest.raises(ValueError):
        nlp.tag_sentence("They book the flight.", "crystal-ball")


@pytest.mark.network
def test_transformer_returns_one_tag_per_word_not_per_piece():
    """Needs your fine-tuned model on disk. The failure this catches: returning
    subword predictions, so 'unhappiness' comes back as three tagged tokens."""
    words = ["They", "booked", "the", "unhappiness", "away", "."]
    tagged = nlp.tag_with_transformer(words)
    assert len(tagged) == len(words)
    assert [t.token for t in tagged] == words
    assert all(t.used_fallback is False for t in tagged)
    assert all(t.confidence is None or 0.0 <= t.confidence <= 1.0 for t in tagged)


@pytest.mark.network
def test_fine_tune_returns_the_hyperparameters_it_actually_used(tiny_treebank):
    """Slow, and six sentences will not produce a usable model — this checks the
    plumbing and the returned record, not the quality. Run your real fine-tune
    from your own script."""
    out = nlp.fine_tune_transformer(
        tiny_treebank,
        tiny_treebank,
        {"base_model": "distilbert-base-uncased", "epochs": 1, "learning_rate": 5e-5,
         "batch_size": 2, "max_length": 32, "seed": 42},
    )
    for key in ["base_model", "epochs", "learning_rate", "seed", "model_path"]:
        assert key in out, f"the run row needs {key} to be reproducible"


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------
def test_token_accuracy_counts_matching_tags():
    gold = ["NOUN", "NOUN", "VERB", "ADJ"]
    pred = ["NOUN", "VERB", "VERB", "ADJ"]
    assert pytest.approx(0.75) == nlp.token_accuracy(gold, pred)


def test_token_accuracy_refuses_mismatched_lengths():
    """A length mismatch is an alignment bug. Zipping to the shorter list turns a
    loud failure into a slightly disappointing number, which is much worse."""
    with pytest.raises(ValueError):
        nlp.token_accuracy(["NOUN", "VERB"], ["NOUN"])


def test_token_accuracy_of_nothing_is_zero():
    assert nlp.token_accuracy([], []) == 0.0


def test_per_tag_f1_is_the_harmonic_mean_of_precision_and_recall():
    gold = ["NOUN", "NOUN", "VERB", "ADJ"]
    pred = ["NOUN", "VERB", "VERB", "ADJ"]
    scores = nlp.per_tag_f1(gold, pred)
    # NOUN: precision 1/1, recall 1/2 -> 0.667.  VERB: precision 1/2, recall 1/1.
    assert pytest.approx(2 / 3, abs=1e-6) == scores["NOUN"]
    assert pytest.approx(2 / 3, abs=1e-6) == scores["VERB"]
    assert pytest.approx(1.0) == scores["ADJ"]


def test_macro_f1_is_dominated_by_the_rare_tag():
    """Nine of ten tokens right, and macro-F1 is under 0.5.

    The one INTJ in the split is never predicted, so its F1 is zero, and macro-F1
    averages it with NOUN as an equal. This is not a bug in the metric — it is
    the metric doing its job, and it is why you report accuracy beside it and say
    which tags moved.
    """
    gold = ["NOUN"] * 9 + ["INTJ"]
    pred = ["NOUN"] * 10
    assert pytest.approx(0.9) == nlp.token_accuracy(gold, pred)
    assert nlp.macro_f1(gold, pred) < 0.6


def test_confusion_matrix_puts_gold_on_rows_and_predictions_on_columns():
    gold = ["NOUN", "NOUN", "VERB"]
    pred = ["NOUN", "VERB", "VERB"]
    cm = nlp.confusion_matrix(gold, pred, labels=["NOUN", "VERB"])
    assert cm.labels == ["NOUN", "VERB"]
    assert cm.matrix == [[1, 1], [0, 1]]
    assert sum(sum(row) for row in cm.matrix) == len(gold)


def test_evaluate_tagger_scores_whatever_tagger_you_hand_it(tiny_treebank):
    """Takes the tagger as an argument so both models are scored by identical
    code. Here the 'tagger' says NOUN to everything, which is the floor."""
    everything_is_a_noun = lambda words: [TaggedToken(token=w, tag="NOUN") for w in words]

    report = nlp.evaluate_tagger(tiny_treebank, everything_is_a_noun, "constant-noun")

    n_tokens = sum(len(s) for s in tiny_treebank)
    n_nouns = sum(1 for s in tiny_treebank for _, tag in s if tag == "NOUN")
    assert report.model == "constant-noun"
    assert report.n_tokens == n_tokens
    assert pytest.approx(n_nouns / n_tokens) == report.accuracy
    assert report.confusion is not None
    assert report.per_tag_f1["NOUN"] > 0.0
    assert report.per_tag_f1.get("VERB", 0.0) == 0.0
