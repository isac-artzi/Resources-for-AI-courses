"""A modular, re-runnable cleaning pipeline built from small pandas functions.

Each function does ONE thing, has a docstring, returns a NEW DataFrame (never
mutates its input), and has a matching unit test. ``run_cleaning`` chains them
in a sensible order. After cleaning, the data should pass the pandera schema in
``src/schema.py``.
"""

import pandas as pd

# The canonical values we normalize text columns to.
VALID_COUNTRIES = ["USA", "CANADA", "BRAZIL", "INDIA"]
VALID_PLANS = ["free", "pro", "enterprise"]

# Ages outside this range are treated as data-entry errors.
MIN_AGE, MAX_AGE = 0, 120


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows, keeping the first occurrence.

    Our synthetic data appended 15 exact copies; duplicates would inflate every
    count and average, so we drop them first.
    """
    return df.drop_duplicates().reset_index(drop=True)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to their intended types.

    ``age`` and ``income`` become numeric (bad values become NaN, handled
    later); ``signup_date`` text becomes real datetimes.
    """
    result = df.copy()
    result["age"] = pd.to_numeric(result["age"], errors="coerce")
    result["income"] = pd.to_numeric(result["income"], errors="coerce")
    result["signup_date"] = pd.to_datetime(result["signup_date"], errors="coerce")
    return result


def standardize_country(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and upper-case the country so 'usa ' == 'USA'."""
    result = df.copy()
    result["country"] = result["country"].str.strip().str.upper()
    return result


def standardize_plan(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and lower-case the plan so 'PRO' == 'pro'."""
    result = df.copy()
    result["plan"] = result["plan"].str.strip().str.lower()
    return result


def replace_invalid_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Turn impossible values into NaN so imputation can replace them.

    Impossible ages (like 999) and negative incomes are clearly errors. We mark
    them missing here, then fill them in ``impute_missing``.
    """
    result = df.copy()
    bad_age = (result["age"] < MIN_AGE) | (result["age"] > MAX_AGE)
    result.loc[bad_age, "age"] = pd.NA
    result.loc[result["income"] < 0, "income"] = pd.NA
    return result


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing ages and incomes with the column median.

    The median is robust to the outliers we just removed. We use the median of
    the *valid* values, then store age as a whole number.
    """
    result = df.copy()
    result["age"] = result["age"].fillna(result["age"].median()).round().astype(int)
    result["income"] = result["income"].fillna(result["income"].median())
    return result


def run_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Run every cleaning step in order and return the cleaned DataFrame.

    Order matters: drop duplicates first, fix types, standardize text, mark
    impossible values missing, then impute.
    """
    df = drop_duplicate_rows(df)
    df = coerce_types(df)
    df = standardize_country(df)
    df = standardize_plan(df)
    df = replace_invalid_with_nan(df)
    df = impute_missing(df)
    return df
