"""Tests for the profiling functions in ``src/profile.py``."""

import pandas as pd

from src.profile import column_profile, numeric_summary, overview


def test_overview_counts_rows_columns_and_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    stats = overview(df)
    assert stats["rows"] == 3
    assert stats["columns"] == 2
    assert stats["duplicate_rows"] == 1  # the (1, "x") row is a duplicate


def test_column_profile_reports_missing_values():
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", "z"]})
    report = column_profile(df)
    a_row = report[report["column"] == "a"].iloc[0]
    assert a_row["null_count"] == 1
    assert a_row["null_pct"] == round(100 / 3, 1)


def test_column_profile_handles_empty_frame():
    report = column_profile(pd.DataFrame({"a": []}))
    assert (report["null_pct"] == 0.0).all()


def test_numeric_summary_only_numeric_columns():
    df = pd.DataFrame({"n": [1.0, 2.0, 3.0], "t": ["a", "b", "c"]})
    summary = numeric_summary(df)
    assert list(summary.columns) == ["n"]


def test_numeric_summary_empty_when_no_numeric():
    df = pd.DataFrame({"t": ["a", "b"]})
    assert numeric_summary(df).empty
