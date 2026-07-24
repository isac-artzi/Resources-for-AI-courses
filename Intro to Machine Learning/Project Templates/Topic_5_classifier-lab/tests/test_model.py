"""Tests for the classifier comparison in ``src/model.py``."""

from src.model import (
    FEATURES,
    evaluate,
    make_models,
    recommend,
    split,
    train_and_compare,
)


def test_make_models_returns_three_pipelines():
    models = make_models()
    assert set(models) == {"kNN", "SVM (RBF)", "Decision Tree"}
    # Every model must scale first (fair, uniform comparison).
    for pipe in models.values():
        assert pipe.steps[0][0] == "scaler"


def test_split_is_stratified_and_reproducible(raw_customers):
    X_train, X_test, y_train, y_test = split(raw_customers)
    assert len(X_train) == 1500
    assert len(X_test) == 500
    assert list(X_train.columns) == FEATURES
    # Both classes present in the test set.
    assert set(y_test.unique()) == {0, 1}


def test_all_models_beat_chance(raw_customers):
    fitted, comparison = train_and_compare(raw_customers)
    assert set(comparison["model"]) == {"kNN", "SVM (RBF)", "Decision Tree"}
    # Every model should be clearly better than a coin flip on this signal.
    assert (comparison["accuracy"] > 0.75).all()
    assert (comparison["roc_auc"] > 0.8).all()
    assert len(fitted) == 3


def test_comparison_is_sorted_and_recommend_matches(raw_customers):
    _, comparison = train_and_compare(raw_customers)
    # Sorted by ROC-AUC, best first.
    assert comparison["roc_auc"].is_monotonic_decreasing
    best = recommend(comparison)
    assert best == comparison.iloc[0]["model"]


def test_evaluate_returns_all_metrics(raw_customers):
    fitted, _ = train_and_compare(raw_customers)
    X_train, X_test, y_train, y_test = split(raw_customers)
    scores = evaluate(fitted["SVM (RBF)"], X_test, y_test)
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"]:
        assert key in scores
