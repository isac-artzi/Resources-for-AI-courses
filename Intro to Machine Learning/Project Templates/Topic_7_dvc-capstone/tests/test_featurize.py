"""Tests for the feature builders in ``src/featurize.py``."""

from src.featurize import (
    NEIGHBORHOODS,
    RAW_NUMERIC,
    baseline_features,
    engineered_features,
)


def test_baseline_uses_raw_numeric_plus_a_code(raw_houses):
    """Baseline is the raw numerics plus a single neighborhood code column."""
    X = baseline_features(raw_houses)
    assert list(X.columns) == RAW_NUMERIC + ["neighborhood_code"]
    assert len(X) == len(raw_houses)


def test_engineered_applies_all_techniques(raw_houses):
    """Engineered features should include one-hot, logs, age bins, and ratios."""
    X = engineered_features(raw_houses)
    # One-hot per neighborhood.
    for name in NEIGHBORHOODS:
        assert f"nb_{name}" in X.columns
    # Log transforms.
    assert {"log_sqft", "log_lot_size"} <= set(X.columns)
    # Ratio features.
    assert {"bed_bath_ratio", "sqft_per_bedroom"} <= set(X.columns)
    # At least one age-bin column.
    assert any(col.startswith("age_") for col in X.columns)


def test_engineered_works_on_a_single_row(raw_houses):
    """The stateless builder must handle one house (for the predict page)."""
    one = raw_houses.head(1)
    X = engineered_features(one)
    assert len(X) == 1
    # A single house belongs to exactly one neighborhood -> one nb_ column is 1.
    nb_cols = [c for c in X.columns if c.startswith("nb_")]
    assert X[nb_cols].sum(axis=1).iloc[0] == 1
