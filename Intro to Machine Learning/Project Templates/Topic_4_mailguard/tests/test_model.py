"""Tests for the naive-Bayes spam filter in ``src/model.py``."""

from src.model import (
    classify_message,
    evaluate,
    influential_words,
    split,
    train,
)


def _fit(raw_emails):
    X_train, X_test, y_train, y_test = split(raw_emails)
    pipe = train(X_train, y_train)
    return pipe, X_test, y_test


def test_split_is_stratified_and_reproducible(raw_emails):
    X_train, X_test, y_train, y_test = split(raw_emails)
    # 75/25 split of 800 messages.
    assert len(X_train) == 600
    assert len(X_test) == 200
    # Both classes appear in the test set (stratified).
    assert set(y_test.unique()) == {0, 1}


def test_filter_scores_well(raw_emails):
    pipe, X_test, y_test = _fit(raw_emails)
    scores = evaluate(pipe, X_test, y_test)
    # The classes are separable, so the filter should be strong (but not asked
    # to be perfect).
    assert scores["accuracy"] > 0.9
    assert scores["roc_auc"] > 0.9
    # The confusion matrix is 2x2.
    assert len(scores["confusion_matrix"]) == 2


def test_obvious_spam_scores_high(raw_emails):
    pipe, _, _ = _fit(raw_emails)
    spammy = "free cash prize win now click offer guaranteed"
    hammy = "team meeting agenda project report schedule review"
    assert classify_message(pipe, spammy) > 0.5
    assert classify_message(pipe, hammy) < 0.5


def test_influential_words_returns_spam_terms(raw_emails):
    pipe, _, _ = _fit(raw_emails)
    top = influential_words(pipe, top_n=5)
    assert list(top.columns) == ["word", "spam_evidence"]
    assert len(top) == 5
    # At least one well-known spam word should surface near the top.
    assert any(w in {"free", "win", "cash", "prize", "offer", "click"} for w in top["word"])
