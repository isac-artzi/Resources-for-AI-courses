"""Tests for the feature-building helpers in ``src/features.py``."""

import pandas as pd

from src.features import chronological_split, make_xy, prepare


def _tiny():
    # Deliberately out of date order to check that prepare() sorts.
    return pd.DataFrame(
        {
            "ticker": ["SYN"] * 4,
            "date": ["2022-01-04", "2022-01-03", "2022-01-06", "2022-01-05"],
            "close": [101.0, 100.0, 103.0, 102.0],
        }
    )


def test_prepare_sorts_and_adds_day_index():
    result = prepare(_tiny())
    assert list(result["t"]) == [0, 1, 2, 3]
    # After sorting, close should be ascending in our tiny example.
    assert list(result["close"]) == [100.0, 101.0, 102.0, 103.0]
    assert pd.api.types.is_datetime64_any_dtype(result["date"])


def test_make_xy_shapes():
    X, y = make_xy(prepare(_tiny()))
    assert list(X.columns) == ["t"]
    assert X.shape == (4, 1)
    assert y.shape == (4,)


def test_chronological_split_keeps_time_order():
    df = _tiny()
    train, test = chronological_split(df, test_frac=0.25)
    assert len(train) == 3
    assert len(test) == 1
    # The test row must be the LAST (most recent) day, not a random one.
    assert test.loc[0, "close"] == 103.0
