"""Tests for the train stage in ``src/train.py``."""

from src.prepare import prepare
from src.train import compare, fit_final, predict_price


def test_engineered_beats_baseline(raw_houses):
    """The whole point: engineered features should win on cross-validated R²."""
    metrics = compare(prepare(raw_houses))
    assert metrics["engineered"]["r2"] > metrics["baseline"]["r2"]
    # On this synthetic data the gain is large and reliable.
    assert metrics["engineered"]["r2"] > 0.85
    assert metrics["r2_gain"] > 0.1


def test_fit_final_bundle_has_columns(raw_houses):
    """The saved bundle carries the training columns for aligned prediction."""
    bundle = fit_final(prepare(raw_houses))
    assert "model" in bundle
    assert isinstance(bundle["columns"], list)
    assert len(bundle["columns"]) > 0


def test_predict_price_returns_a_positive_number(raw_houses):
    """Scoring one house should give a sensible positive price."""
    prepared = prepare(raw_houses)
    bundle = fit_final(prepared)
    price = predict_price(bundle, prepared.head(1))
    assert price > 0
