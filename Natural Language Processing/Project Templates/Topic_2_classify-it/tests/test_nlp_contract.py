"""YOUR TO-DO LIST, WRITTEN AS TESTS.

Every test here fails on a fresh fork with NotImplementedError. Each one
describes a behaviour that ``api/nlp.py`` must have. Work top to bottom:

    pytest -m contract -x                      # stop at the first thing left to do
    pytest -m "contract and not model" -x      # skip anything needing an artifact

These are deliberately loose about *how* — they check the contract, not your
algorithm. Nothing here says which n-gram range to use, how many epochs to train
for, or whether you calibrate with Platt scaling or isotonic regression. They do
say that a batch comes back in order, that a probability is the probability of
the label you actually predicted, and that recall on an imbalanced corpus is not
accuracy.

Passing them is necessary, not sufficient: the report and the Baseline vs.
Transformer tab are where you show you understood what you built.
"""
from __future__ import annotations

import pytest

from api import nlp
from shared.schemas import LabeledExample

pytestmark = pytest.mark.contract


def _corpus(n_urgent: int = 30, n_routine: int = 70) -> list[LabeledExample]:
    """A small, deliberately imbalanced two-label corpus."""
    return [
        LabeledExample(text=f"urgent message number {i}", label="urgent")
        for i in range(n_urgent)
    ] + [
        LabeledExample(text=f"routine question number {i}", label="routine")
        for i in range(n_routine)
    ]


# ---------------------------------------------------------------------------
# 1. Loading the corpus
# ---------------------------------------------------------------------------
def test_load_labeled_dataset_reads_a_csv_of_text_and_label(tiny_corpus_csv):
    examples = nlp.load_labeled_dataset(tiny_corpus_csv)
    assert examples, "loader returned nothing"
    assert all(isinstance(e, LabeledExample) for e in examples)
    assert {e.label for e in examples} == {"urgent", "routine"}


def test_load_labeled_dataset_drops_rows_with_empty_text(tiny_corpus_csv):
    # The fixture file has 12 data rows, one of which has a label but no text.
    # A blank message is not an example; keeping it teaches the model that the
    # empty string means "routine".
    examples = nlp.load_labeled_dataset(tiny_corpus_csv)
    assert len(examples) == 11
    assert all(e.text.strip() for e in examples)


def test_load_labeled_dataset_preserves_file_order(tiny_corpus_csv):
    examples = nlp.load_labeled_dataset(tiny_corpus_csv)
    assert examples[0].text.startswith("my order never arrived")


# ---------------------------------------------------------------------------
# 2. Validating the corpus — the single-label, two-value rule
# ---------------------------------------------------------------------------
def test_validate_dataset_describes_the_corpus():
    schema = nlp.validate_dataset(_corpus(n_urgent=30, n_routine=70))
    assert set(schema.labels) == {"urgent", "routine"}
    assert schema.positive_label in schema.labels
    assert schema.class_counts == {"urgent": 30, "routine": 70}
    assert schema.n_rows == 100
    assert schema.minority_share() == pytest.approx(0.30)


def test_validate_dataset_rejects_a_third_label():
    # Multiclass is out of scope for this product. Fail here, loudly, rather
    # than training something whose reported F1 nobody can interpret.
    bad = _corpus(5, 5) + [LabeledExample(text="spam text", label="spam")]
    with pytest.raises(ValueError):
        nlp.validate_dataset(bad)


def test_validate_dataset_rejects_a_single_label():
    with pytest.raises(ValueError):
        nlp.validate_dataset(_corpus(n_urgent=10, n_routine=0))


def test_validate_dataset_rejects_an_empty_corpus():
    with pytest.raises(ValueError):
        nlp.validate_dataset([])


# ---------------------------------------------------------------------------
# 3. The held-out split
# ---------------------------------------------------------------------------
def test_stratified_split_is_disjoint_and_complete():
    corpus = _corpus(30, 70)
    train, held_out = nlp.stratified_split(corpus, test_size=0.2, seed=42)
    assert len(train) + len(held_out) == len(corpus)
    train_texts = {e.text for e in train}
    held_texts = {e.text for e in held_out}
    assert not (train_texts & held_texts), "a row appeared in both splits — that is a leak"


def test_stratified_split_preserves_class_balance():
    corpus = _corpus(30, 70)
    _, held_out = nlp.stratified_split(corpus, test_size=0.2, seed=42)
    urgent = sum(1 for e in held_out if e.label == "urgent")
    # 30% urgent overall, so roughly 30% of the held-out rows should be urgent.
    # Loose bounds on purpose: rounding is your business, stratification is not.
    assert 0.2 <= urgent / len(held_out) <= 0.4


def test_stratified_split_is_deterministic_given_a_seed():
    corpus = _corpus(30, 70)
    a, _ = nlp.stratified_split(corpus, test_size=0.2, seed=7)
    b, _ = nlp.stratified_split(corpus, test_size=0.2, seed=7)
    assert [e.text for e in a] == [e.text for e in b], (
        "same seed, same corpus, different split — you cannot compare two models "
        "that were graded on different exams"
    )


# ---------------------------------------------------------------------------
# 4. Metrics. Read this section before you train anything.
# ---------------------------------------------------------------------------
def test_compute_metrics_is_perfect_when_the_predictions_are():
    y = ["urgent", "routine", "urgent", "routine"]
    m = nlp.compute_metrics(y, list(y), positive_label="urgent")
    assert m.accuracy == pytest.approx(1.0)
    assert m.precision == pytest.approx(1.0)
    assert m.recall == pytest.approx(1.0)
    assert m.f1 == pytest.approx(1.0)
    assert m.n_examples == 4
    assert m.support_positive == 2


def test_compute_metrics_counts_the_confusion_matrix_correctly():
    #            predicted
    #            urgent  routine
    # urgent        2       2      -> recall    2/4 = 0.5
    # routine       1       3      -> precision 2/3
    y_true = ["urgent"] * 4 + ["routine"] * 4
    y_pred = ["urgent", "urgent", "routine", "routine", "urgent"] + ["routine"] * 3
    m = nlp.compute_metrics(y_true, y_pred, positive_label="urgent")
    assert m.accuracy == pytest.approx(5 / 8)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(0.5)
    assert m.f1 == pytest.approx(2 * (2 / 3) * 0.5 / ((2 / 3) + 0.5))


def test_accuracy_lies_on_an_imbalanced_corpus():
    """THE TEST THIS WHOLE PRODUCT EXISTS TO MAKE YOU READ.

    Ninety routine messages, ten urgent ones, and a model that answers
    "routine" every single time. It scores 0.90 accuracy. It has never once
    found the thing the customer-operations team actually cares about.
    """
    y_true = ["routine"] * 90 + ["urgent"] * 10
    y_pred = ["routine"] * 100
    m = nlp.compute_metrics(y_true, y_pred, positive_label="urgent")
    assert m.accuracy == pytest.approx(0.90)
    assert m.recall == pytest.approx(0.0)
    assert m.f1 == pytest.approx(0.0)
    # Precision is 0/0 here. Return 0.0 rather than raising or returning NaN —
    # NaN propagates silently into your report and into the runs table.
    assert m.precision == pytest.approx(0.0)


def test_compute_metrics_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        nlp.compute_metrics(["urgent", "routine"], ["urgent"], positive_label="urgent")


# ---------------------------------------------------------------------------
# 5. Training. The fast half of the contract: validate before you spend compute.
# ---------------------------------------------------------------------------
def test_fit_baseline_refuses_a_corpus_that_is_not_binary():
    bad = _corpus(5, 5) + [LabeledExample(text="third class", label="spam")]
    with pytest.raises(ValueError):
        nlp.fit_baseline(bad, {"C": 1.0})


def test_fine_tune_transformer_refuses_a_corpus_that_is_not_binary():
    # This check must happen before the checkpoint download. Finding out after
    # twenty minutes of training is the expensive version of the same bug.
    bad = _corpus(5, 5) + [LabeledExample(text="third class", label="spam")]
    with pytest.raises(ValueError):
        nlp.fine_tune_transformer(bad, bad, {"epochs": 1})


@pytest.mark.model
def test_fit_baseline_returns_reproducible_run_metadata():
    train, held_out = nlp.stratified_split(_corpus(40, 60), test_size=0.2, seed=42)
    meta = nlp.fit_baseline(train, {"C": 1.0, "ngram_range": [1, 2], "min_df": 1})
    assert meta["model_kind"] == "baseline"
    assert meta["model_version"], "a run without a version cannot be joined to its predictions"
    hp = meta["hyperparameters"]
    assert hp, "record the hyperparameters — a run you cannot reproduce is an anecdote"
    # The vectorizer settings are hyperparameters too, and they are the ones
    # students forget. If your dict has C but no n-gram range, you cannot
    # reproduce the feature matrix.
    assert any("gram" in k or "vector" in k or "tfidf" in k for k in hp), (
        "hyperparameters must describe the vectorizer, not only the classifier"
    )
    assert meta.get("n_train") == len(train)
    assert held_out, "sanity: the split gave you something to evaluate on"


@pytest.mark.model
def test_fine_tune_transformer_returns_run_metadata():
    """Slow and network-bound: it downloads a checkpoint and trains.

    Point it at a tiny config (1 epoch, small max_length) while you are
    developing. It is here so the metadata shape is pinned, not to measure
    accuracy on 100 synthetic rows.
    """
    train, held_out = nlp.stratified_split(_corpus(40, 60), test_size=0.2, seed=42)
    meta = nlp.fine_tune_transformer(
        train,
        held_out,
        {"checkpoint": "distilbert-base-uncased", "epochs": 1, "max_length": 64},
    )
    assert meta["model_kind"] == "transformer"
    assert meta["model_version"]
    hp = meta["hyperparameters"]
    for key in ("checkpoint", "learning_rate", "epochs", "seed"):
        assert key in hp, f"record {key} — it changes the result"


# ---------------------------------------------------------------------------
# 6. Serving
# ---------------------------------------------------------------------------
def test_label_schema_is_binary_and_documented():
    schema = nlp.label_schema()
    assert len(set(schema.labels)) == 2
    assert schema.positive_label in schema.labels
    assert schema.dataset_source, (
        "GET /schema is where a reviewer finds your corpus. Cite it."
    )
    assert schema.label_definitions, (
        "one sentence per label — the Build Steps ask for label definitions"
    )


def test_load_model_rejects_an_unknown_kind():
    with pytest.raises(ValueError):
        nlp.load_model("definitely-not-a-model-kind")


@pytest.mark.model
def test_load_model_returns_the_same_cached_object():
    first = nlp.load_model("baseline")
    second = nlp.load_model("baseline")
    assert first is second, (
        "load_model must cache. Deserializing a model per request is why a "
        "correct service feels broken on the free plan."
    )


@pytest.mark.model
def test_score_texts_returns_one_distribution_per_text():
    scores = nlp.score_texts(["my order never arrived", "thanks"], "baseline")
    assert len(scores) == 2
    labels = set(nlp.label_schema().labels)
    for s in scores:
        assert set(s) == labels, "score every label in the schema, by its name"
        assert sum(s.values()) == pytest.approx(1.0, abs=1e-6)
        assert all(0.0 <= v <= 1.0 for v in s.values())


@pytest.mark.model
def test_calibrate_keeps_the_shape_and_the_ordering():
    texts = ["my order never arrived", "thanks", "any update on this", "how do i log in"]
    raw = nlp.score_texts(texts, "baseline")
    cal = nlp.calibrate(raw, "baseline")
    assert len(cal) == len(raw)
    pos = nlp.label_schema().positive_label
    for c in cal:
        assert set(c) == set(raw[0])
        assert sum(c.values()) == pytest.approx(1.0, abs=1e-6)
    # Platt scaling and isotonic regression are both monotonic: calibration
    # rescales confidence, it never reorders which input looks more positive.
    raw_order = sorted(range(len(raw)), key=lambda i: raw[i][pos])
    cal_order = sorted(range(len(cal)), key=lambda i: cal[i][pos])
    assert raw_order == cal_order, "calibration reordered your inputs — that is a bug"


@pytest.mark.model
def test_predict_reports_the_probability_of_the_label_it_predicted():
    result = nlp.predict("my order never arrived and nobody replied", "baseline")
    assert result.label in nlp.label_schema().labels
    assert result.model_kind == "baseline"
    assert result.model_version, "which artifact answered? the log is useless without it"
    assert result.text_sha256 == nlp.sha256_text(
        "my order never arrived and nobody replied"
    )
    # In a two-class problem the winning label's probability cannot be below 0.5.
    # If this fails you are returning the positive class's probability regardless
    # of what you predicted, and the UI is showing "routine, 12% confident".
    assert 0.5 <= result.probability <= 1.0


@pytest.mark.model
def test_predict_batch_is_one_result_per_input_in_order():
    texts = ["refund me now", "thanks that worked", "i was charged twice"]
    results = nlp.predict_batch(texts, "baseline")
    assert len(results) == len(texts)
    for text, r in zip(texts, results):
        assert r.text_sha256 == nlp.sha256_text(text), (
            "results came back in a different order than the inputs — the caller "
            "zips these against their own rows and will mislabel every one"
        )


@pytest.mark.model
def test_predict_batch_agrees_with_predict():
    text = "i was charged twice and nobody has replied"
    one = nlp.predict(text, "baseline")
    many = nlp.predict_batch([text, "thanks"], "baseline")
    assert many[0].label == one.label
    assert many[0].probability == pytest.approx(one.probability, abs=1e-6), (
        "the single and batch paths disagree — share one code path so they cannot"
    )


def test_predict_batch_of_nothing_is_not_an_error():
    assert nlp.predict_batch([], "baseline") == []
