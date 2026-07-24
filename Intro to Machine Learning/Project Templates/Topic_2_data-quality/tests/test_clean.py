"""Unit tests for the cleaning pipeline in ``src/clean.py``.

Each cleaning function does one thing, so each gets a small, focused test. We
also test ``run_cleaning`` end-to-end on the real messy CSV.
"""

import pandas as pd

from src.clean import (
    coerce_types,
    drop_duplicate_rows,
    impute_missing,
    replace_invalid_with_nan,
    run_cleaning,
    standardize_country,
    standardize_plan,
)


def test_drop_duplicate_rows_removes_exact_copies():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    result = drop_duplicate_rows(df)
    assert len(result) == 2


def test_coerce_types_makes_numbers_and_dates():
    df = pd.DataFrame(
        {"age": ["40", "bad"], "income": ["1000", "2000"], "signup_date": ["2023-01-01", "nope"]}
    )
    result = coerce_types(df)
    assert pd.api.types.is_numeric_dtype(result["age"])
    assert pd.isna(result.loc[1, "age"])  # "bad" became NaN
    assert pd.api.types.is_datetime64_any_dtype(result["signup_date"])


def test_standardize_country_upper_and_trimmed():
    df = pd.DataFrame({"country": [" usa ", "Canada"]})
    result = standardize_country(df)
    assert list(result["country"]) == ["USA", "CANADA"]


def test_standardize_plan_lower_and_trimmed():
    df = pd.DataFrame({"plan": [" PRO", "Free "]})
    result = standardize_plan(df)
    assert list(result["plan"]) == ["pro", "free"]


def test_replace_invalid_with_nan():
    df = pd.DataFrame({"age": [40, 999], "income": [1000.0, -5.0]})
    result = replace_invalid_with_nan(df)
    assert pd.isna(result.loc[1, "age"])
    assert pd.isna(result.loc[1, "income"])


def test_impute_missing_fills_with_median():
    df = pd.DataFrame({"age": [10.0, None, 30.0], "income": [100.0, None, 300.0]})
    result = impute_missing(df)
    assert result["age"].isna().sum() == 0
    assert result["income"].isna().sum() == 0
    assert result["age"].dtype == int


def test_run_cleaning_produces_clean_frame(raw_signups):
    cleaned = run_cleaning(raw_signups)
    # No missing values remain in the columns we impute.
    assert cleaned["age"].isna().sum() == 0
    assert cleaned["income"].isna().sum() == 0
    # Duplicates removed, so signup_id is unique again.
    assert cleaned["signup_id"].is_unique
    # Text columns standardized.
    assert set(cleaned["country"]).issubset({"USA", "CANADA", "BRAZIL", "INDIA"})
    assert set(cleaned["plan"]).issubset({"free", "pro", "enterprise"})
    # Ages back in the plausible range.
    assert cleaned["age"].between(0, 120).all()
    assert (cleaned["income"] >= 0).all()
