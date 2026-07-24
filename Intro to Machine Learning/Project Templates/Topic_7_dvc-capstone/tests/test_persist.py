"""Tests for saving and loading derived CSV files."""

from datetime import date

import pandas as pd

from src.persist import derived_filename, load_derived, save_derived


def test_derived_filename_includes_date():
    name = derived_filename("top_customers", on=date(2024, 1, 15))
    assert name == "cleaned_top_customers_20240115.csv"


def test_save_and_load_roundtrip(tmp_path):
    """A DataFrame saved and re-loaded should be identical."""
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = save_derived(df, "sample", derived_dir=tmp_path, on=date(2024, 1, 15))

    assert path.exists()
    assert path.name == "cleaned_sample_20240115.csv"

    reloaded = load_derived(path)
    pd.testing.assert_frame_equal(df, reloaded)
