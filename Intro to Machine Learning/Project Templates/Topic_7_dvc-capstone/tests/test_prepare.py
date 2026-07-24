"""Tests for the prepare stage in ``src/prepare.py``."""

import pandas as pd

from src.prepare import prepare


def test_prepare_keeps_valid_rows(raw_houses):
    """Clean synthetic data should survive preparation unchanged in count."""
    prepared = prepare(raw_houses)
    assert len(prepared) == len(raw_houses)
    assert "price" in prepared.columns


def test_prepare_drops_duplicates_and_bad_prices():
    """Duplicate rows and non-positive prices should be removed."""
    df = pd.DataFrame(
        {
            "neighborhood": ["Ashby", "Ashby", "Brook"],
            "sqft": [1000, 1000, 1200],
            "lot_size": [5000, 5000, 6000],
            "year_built": [2000, 2000, 1990],
            "bedrooms": [3, 3, 4],
            "bathrooms": [2, 2, 2],
            "garage": [1, 1, 0],
            "condition": [3, 3, 4],
            "price": [250000, 250000, -5],  # a duplicate and an invalid price
        }
    )
    prepared = prepare(df)
    # One duplicate removed, one invalid price removed -> a single row remains.
    assert len(prepared) == 1
    assert (prepared["price"] > 0).all()
