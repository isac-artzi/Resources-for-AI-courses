"""Profile a dataset: measure its quality before you trust it.

Profiling answers the first questions you should ask about any new data:
how many rows, what types, how many missing values, how many duplicates, and
what do the numbers look like? These functions return plain DataFrames/dicts so
the Streamlit UI can render them and the tests can check them.
"""

import pandas as pd


def overview(df: pd.DataFrame) -> dict:
    """Return headline counts: rows, columns, and exact-duplicate rows."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column quality report.

    One row per column with its dtype, how many values are missing, the missing
    percentage, and how many distinct values it has.
    """
    n = len(df)
    report = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "null_count": df.isnull().sum(),
            "null_pct": (df.isnull().mean() * 100).round(1),
            "distinct_values": df.nunique(),
        }
    )
    report.index.name = "column"
    # Guard against an empty DataFrame (n == 0) so we never divide by zero.
    if n == 0:
        report["null_pct"] = 0.0
    return report.reset_index()


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return min/max/mean/etc. for the numeric columns (empty if none)."""
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    return numeric.describe()
