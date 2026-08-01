"""YOUR TO-DO LIST, WRITTEN AS TESTS.

Every test here fails on a fresh fork with NotImplementedError. Each one
describes a behaviour that ``api/nlp.py`` must have. Work top to bottom:

    pytest -m contract -x                        # stop at the first thing to do
    pytest -m "contract and not network"         # offline: metrics and baseline
    pytest -m "not train"                        # skip fine-tuning

These are deliberately loose about *how* — they check the contract, not your
algorithm. Passing them is necessary, not sufficient: the report, the bias audit
and the model card are where you show you understood what you built.

Notice how much of this file needs no model at all. The evaluation functions are
pure, so you can write and check them on eight hand-made examples in a second,
before you have spent an hour training anything. Do that. A metric you only ever
ran on real output is a metric you have never actually tested.
"""
from __future__ import annotations

import os

import pytest

from api import nlp
from shared.schemas import AspectSentiment

pytestmark = pytest.mark.contract


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
@pytest.mark.network
def test_load_reviews_returns_labelled_text():
    rows = nlp.load_reviews("test", limit=8)
    assert 0 < len(rows) <= 8
    for row in rows:
        assert isinstance(row["text"], str) and row["text"].strip()
        assert row["label"] in {"negative", "positive"}


@pytest.mark.network
def test_train_and_test_splits_do_not_overlap():
    # Leakage is silent and it inflates every number you go on to report. Check
    # it once, here, rather than wondering later why the test score is so good.
    train = {r["text"] for r in nlp.load_reviews("train", limit=200)}
    test = {r["text"] for r in nlp.load_reviews("test", limit=200)}
    assert not (train & test)


# ---------------------------------------------------------------------------
# 2. The TF-IDF baseline
#    No network, no GPU, no excuse to skip it. This is the number the
#    transformer has to beat.
# ---------------------------------------------------------------------------
POS = [
    "a wonderful film, warm and funny",
    "excellent performances and a beautiful score",
    "wonderful, moving, and beautifully shot",
    "an excellent, warm, funny film",
]
NEG = [
    "a terrible film, dull and lifeless",
    "awful performances and a tedious script",
    "terrible, dull, and badly shot",
    "an awful, lifeless, tedious film",
]
TINY_TRAIN = [{"text": t, "label": "positive"} for t in POS] + [
    {"text": t, "label": "negative"} for t in NEG
]


def test_baseline_fits_and_predicts_in_order():
    model = nlp.fit_baseline(TINY_TRAIN)
    preds = nlp.baseline_predict(model, [POS[0], NEG[0]])

    assert len(preds) == 2
    for p in preds:
        assert p.model_name == "tfidf-baseline"
        assert 0.0 <= p.probability_positive <= 1.0
        assert 0.5 <= p.confidence <= 1.0
        # confidence is the probability of the label you actually predicted
        assert p.confidence == pytest.approx(
            max(p.probability_positive, 1 - p.probability_positive), abs=1e-6
        )
    # Predicting the training examples back is the lowest bar there is. If this
    # fails, the labels are crossed somewhere.
    assert preds[0].label == "positive"
    assert preds[1].label == "negative"


# ---------------------------------------------------------------------------
# 3. The transformer
# ---------------------------------------------------------------------------
def test_fine_tune_rejects_an_unknown_base_model_before_downloading_anything():
    with pytest.raises(ValueError):
        nlp.fine_tune_transformer(
            TINY_TRAIN,
            TINY_TRAIN,
            {"base_model": "definitely-not-a-real-checkpoint-id-42", "epochs": 1},
        )


@pytest.mark.train
@pytest.mark.skipif(
    not os.environ.get("MOODLENS_RUN_TRAINING"),
    reason="set MOODLENS_RUN_TRAINING=1 to actually fine-tune (slow)",
)
def test_fine_tune_returns_the_version_it_wrote(tmp_path):
    version = nlp.fine_tune_transformer(
        TINY_TRAIN,
        TINY_TRAIN,
        {
            "base_model": "distilbert-base-uncased",
            "output_dir": str(tmp_path / "moodlens-test"),
            "model_version": "moodlens-test",
            "epochs": 1,
            "batch_size": 4,
            "max_length": 64,
            "learning_rate": 5e-5,
            "seed": 13,
        },
    )
    assert version == "moodlens-test"
    assert (tmp_path / "moodlens-test").exists()


@pytest.mark.network
def test_classifier_is_cached_not_reloaded_per_call():
    # A per-request load is not slow, it is broken: on the free plan the Score
    # Text tab will time out while your code looks correct.
    assert nlp.load_classifier() is nlp.load_classifier()


def test_calibrator_is_a_monotonic_map_into_zero_one():
    scores = [0.05, 0.2, 0.4, 0.55, 0.7, 0.85, 0.95, 0.99]
    labels = ["negative"] * 4 + ["positive"] * 4
    calibrate = nlp.fit_calibrator(scores, labels)

    values = [calibrate(s) for s in scores]
    assert all(0.0 <= v <= 1.0 for v in values)
    # Calibration may change how confident the model is. It may not change which
    # side of the line an item lands on relative to another item.
    assert values == sorted(values)


@pytest.mark.network
def test_predict_sentiment_fills_every_field():
    p = nlp.predict_sentiment(
        "An absolute masterpiece — I loved every second of it and I will watch it again."
    )
    assert p.label in {"negative", "positive"}
    assert 0.0 <= p.probability_positive <= 1.0
    assert 0.0 <= p.raw_probability_positive <= 1.0
    assert p.confidence == pytest.approx(
        max(p.probability_positive, 1 - p.probability_positive), abs=1e-6
    )
    assert isinstance(p.calibrated, bool)
    # If this one fails while the others pass, your label mapping is inverted:
    # index 1 is not always "positive". Read it off the model config.
    assert p.label == "positive"
    # The label must follow the CALIBRATED probability, not the raw one.
    assert (p.probability_positive >= 0.5) == (p.label == "positive")


@pytest.mark.network
def test_predict_sentiment_batch_preserves_order_and_length():
    texts = [
        "I loved it, easily the best thing I have seen this year.",
        "Dreadful. I walked out after forty minutes.",
        "I loved it, easily the best thing I have seen this year.",
    ]
    preds = nlp.predict_sentiment_batch(texts)
    assert len(preds) == 3
    # Same text in, same answer out. If items 0 and 2 disagree, something in the
    # batch path (padding, truncation, sorting by length) is leaking between rows.
    assert preds[0].label == preds[2].label
    assert preds[0].probability_positive == pytest.approx(
        preds[2].probability_positive, abs=1e-4
    )
    assert preds[0].label != preds[1].label


@pytest.mark.network
def test_empty_batch_is_an_empty_list_not_an_error():
    assert nlp.predict_sentiment_batch([]) == []


# ---------------------------------------------------------------------------
# 4. Aspects
# ---------------------------------------------------------------------------
MIXED_REVIEW = (
    "The performances are superb and the cast is perfectly chosen, but the plot "
    "is a mess and the story falls apart in the final act."
)


@pytest.mark.network
def test_extract_aspects_returns_one_entry_per_documented_aspect():
    out = nlp.extract_aspects(MIXED_REVIEW)
    assert [a.aspect for a in out] == list(nlp.ASPECTS)
    for a in out:
        assert isinstance(a, AspectSentiment)
        assert 0.0 <= a.score <= 1.0


@pytest.mark.network
def test_aspect_evidence_is_quoted_from_the_input():
    # Not paraphrased, not cleaned up, not invented. The point of the field is
    # that a human can find the words in the review.
    for a in nlp.extract_aspects(MIXED_REVIEW):
        for snippet in a.evidence:
            assert snippet in MIXED_REVIEW


@pytest.mark.network
def test_an_aspect_the_review_never_mentions_is_reported_as_absent():
    out = nlp.extract_aspects("The performances are superb and the cast is perfect.")
    by_aspect = {a.aspect: a for a in out}
    assert by_aspect["plot"].label == "not_mentioned"
    assert by_aspect["plot"].evidence == []


@pytest.mark.network
def test_aspects_can_disagree_with_each_other():
    # The whole reason aspect-based sentiment exists. An aspect step that never
    # disagrees with the document label is repeating the document classifier.
    labels = {a.aspect: a.label for a in nlp.extract_aspects(MIXED_REVIEW)}
    assert labels["acting"] == "positive"
    assert labels["plot"] == "negative"


# ---------------------------------------------------------------------------
# 5. Evaluation — pure functions, no model, milliseconds to run
# ---------------------------------------------------------------------------
def test_perfect_predictions_score_one(prediction_factory):
    preds = [prediction_factory(0.99), prediction_factory(0.01), prediction_factory(0.98)]
    gold = ["positive", "negative", "positive"]
    m = nlp.evaluate_documents(preds, gold, model_name="transformer")

    assert m.accuracy == pytest.approx(1.0)
    assert m.macro_f1 == pytest.approx(1.0)
    assert m.n == 3
    assert m.model_name == "transformer"
    assert m.per_class["positive"].support == 2
    assert m.per_class["negative"].support == 1


def test_confusion_matrix_is_gold_by_predicted(prediction_factory):
    # Two reviews, both called positive; one of them is really negative.
    preds = [prediction_factory(0.9), prediction_factory(0.8)]
    gold = ["negative", "positive"]
    m = nlp.evaluate_documents(preds, gold, model_name="transformer")

    assert m.labels == ["negative", "positive"]
    assert m.confusion_matrix == [[0, 1], [0, 1]]  # [gold][predicted]


def test_macro_f1_is_not_accuracy_on_an_imbalanced_split(prediction_factory):
    # Nine positives, one negative, and a model that says positive every time.
    # Accuracy 0.9 looks fine. Macro-F1 is the number that notices the model
    # never gets the small class right — which is exactly why it is the one the
    # assignment asks for.
    preds = [prediction_factory(0.9) for _ in range(10)]
    gold = ["positive"] * 9 + ["negative"]
    m = nlp.evaluate_documents(preds, gold, model_name="transformer")

    assert m.accuracy == pytest.approx(0.9)
    assert m.macro_f1 < 0.6
    assert m.per_class["negative"].recall == pytest.approx(0.0)


def test_mismatched_lengths_raise_rather_than_produce_a_plausible_number(
    prediction_factory,
):
    with pytest.raises(ValueError):
        nlp.evaluate_documents(
            [prediction_factory(0.9)], ["positive", "negative"], model_name="transformer"
        )


def test_calibration_bins_report_what_happened_not_what_was_claimed(prediction_factory):
    # Ten reviews the model is 85% sure are positive. Five of them are not.
    # A correct reliability curve says: predicted 0.85, observed 0.5.
    preds = [prediction_factory(0.85) for _ in range(10)]
    gold = ["positive"] * 5 + ["negative"] * 5
    m = nlp.evaluate_documents(preds, gold, model_name="transformer", n_calibration_bins=10)

    populated = [b for b in m.calibration_bins if b.count > 0]
    assert len(populated) == 1
    assert sum(b.count for b in populated) == 10
    assert populated[0].mean_predicted == pytest.approx(0.85, abs=1e-6)
    assert populated[0].observed_positive_rate == pytest.approx(0.5, abs=1e-6)


def test_roc_points_sweep_the_threshold(prediction_factory):
    preds = [prediction_factory(p) for p in (0.1, 0.4, 0.6, 0.9)]
    gold = ["negative", "negative", "positive", "positive"]
    m = nlp.evaluate_documents(preds, gold, model_name="transformer")

    assert len(m.roc_points) >= 3
    for point in m.roc_points:
        assert 0.0 <= point.fpr <= 1.0
        assert 0.0 <= point.tpr <= 1.0
    # Perfectly separable scores.
    assert m.roc_auc == pytest.approx(1.0, abs=1e-6)


def _aspects(**labels) -> list[AspectSentiment]:
    """One AspectSentiment per documented aspect, defaulting to not_mentioned."""
    return [
        AspectSentiment(
            aspect=a, label=labels.get(a, "not_mentioned"), score=0.9, evidence=[]
        )
        for a in nlp.ASPECTS
    ]


def test_per_aspect_metrics_cover_every_aspect_even_unannotated_ones():
    first, second = nlp.ASPECTS[0], nlp.ASPECTS[1]
    predicted = [
        _aspects(**{first: "positive", second: "negative"}),
        _aspects(**{first: "positive", second: "negative"}),
    ]
    gold = [
        {first: "positive", second: "negative"},
        {first: "positive", second: "negative"},
    ]
    out = nlp.evaluate_aspects(predicted, gold)

    by_aspect = {m.aspect: m for m in out}
    assert set(by_aspect) == set(nlp.ASPECTS)
    assert by_aspect[first].f1 == pytest.approx(1.0)
    assert by_aspect[first].n_evaluated == 2
    # The third aspect has no gold annotation anywhere. It is still reported,
    # with zero support — a missing row reads like a score of zero, and those
    # are very different findings.
    assert by_aspect[nlp.ASPECTS[2]].n_evaluated == 0
    assert by_aspect[nlp.ASPECTS[2]].support == 0


# ---------------------------------------------------------------------------
# 6. Slices — the plumbing behind the Bias Audit tab
# ---------------------------------------------------------------------------
SHORT = {"text": "Loved it.", "label": "positive"}
LONG = {"text": ("wonderful " * 400).strip(), "label": "positive"}


def test_slice_of_puts_short_and_long_reviews_in_different_buckets():
    short_bucket = nlp.slice_of(SHORT, "review_length")
    long_bucket = nlp.slice_of(LONG, "review_length")
    buckets = nlp.SLICE_SPECS["review_length"]["buckets"]
    assert short_bucket in buckets and long_bucket in buckets
    assert short_bucket != long_bucket


def test_slice_of_rejects_a_slice_you_never_defined():
    with pytest.raises(ValueError):
        nlp.slice_of(SHORT, "star_sign")


def test_evaluate_slices_partitions_the_split(prediction_factory):
    records = [SHORT, LONG, dict(SHORT, label="negative"), LONG]
    preds = [
        prediction_factory(0.9),
        prediction_factory(0.9),
        prediction_factory(0.9),  # wrong: this one is really negative
        prediction_factory(0.9),
    ]
    out = nlp.evaluate_slices(records, preds, "review_length")

    assert out, "at least one bucket must come back"
    assert {m.slice_name for m in out} == {"review_length"}
    # Every record lands in exactly one bucket, so the counts partition the split.
    assert sum(m.n for m in out) == len(records)
    assert all(m.bucket in nlp.SLICE_SPECS["review_length"]["buckets"] for m in out)
    # review_length is measured, not guessed.
    assert all(m.observed for m in out)
    for m in out:
        assert 0.0 <= m.accuracy <= 1.0
        assert 0.0 <= m.macro_f1 <= 1.0
