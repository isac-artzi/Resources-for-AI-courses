"""Tests for the pandas cleaning helpers."""

import pandas as pd

from src.clean import drop_duplicate_customers, parse_invoice_dates, standardize_country


def test_standardize_country_trims_and_uppercases():
    df = pd.DataFrame({"country": [" usa ", "Canada", "brazil"]})
    result = standardize_country(df)
    assert result["country"].tolist() == ["USA", "CANADA", "BRAZIL"]


def test_drop_duplicate_customers_removes_exact_duplicates():
    df = pd.DataFrame(
        {
            "customer_id": [1, 1, 2],
            "first_name": ["Ava", "Ava", "Ben"],
        }
    )
    result = drop_duplicate_customers(df)
    assert len(result) == 2


def test_parse_invoice_dates_returns_datetimes():
    df = pd.DataFrame({"invoice_date": ["2024-01-05", "2024-02-20"]})
    result = parse_invoice_dates(df)
    assert pd.api.types.is_datetime64_any_dtype(result["invoice_date"])


def test_clean_functions_do_not_mutate_input():
    """Cleaning should return a new DataFrame, leaving the original untouched."""
    df = pd.DataFrame({"country": [" usa "]})
    standardize_country(df)
    assert df["country"].iloc[0] == " usa "  # unchanged
