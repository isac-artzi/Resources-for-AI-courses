"""Save derived DataFrames as versioned CSV files.

Why CSV, and why versioned?

Streamlit Community Cloud's file system is *ephemeral*: anything the app
writes disappears when the app restarts or redeploys, and writes to the
bundled SQLite file are not reliable. The course convention is therefore:

    read from SQLite  ->  clean/join in pandas  ->  write derived state to CSV

We stamp each file with the date (``cleaned_top_customers_20240115.csv``) so
you can see when it was produced and keep older versions for comparison.
"""

from datetime import date
from pathlib import Path

import pandas as pd

from src.paths import DERIVED_DIR


def derived_filename(name: str, on: date | None = None) -> str:
    """Build a dated file name like ``cleaned_<name>_<YYYYMMDD>.csv``."""
    stamp = (on or date.today()).strftime("%Y%m%d")
    return f"cleaned_{name}_{stamp}.csv"


def save_derived(
    df: pd.DataFrame,
    name: str,
    derived_dir: Path = DERIVED_DIR,
    on: date | None = None,
) -> Path:
    """Write ``df`` to a dated CSV under ``data/derived`` and return its path.

    Parameters
    ----------
    df:
        The cleaned/derived DataFrame to save.
    name:
        A short label describing the contents (e.g. ``"top_customers"``).
    derived_dir:
        Where to write. Defaults to the project's ``data/derived`` folder.
    on:
        Optional date for the filename stamp (defaults to today). Passing a
        fixed date makes tests predictable.
    """
    derived_dir.mkdir(parents=True, exist_ok=True)
    out_path = derived_dir / derived_filename(name, on)
    # index=False keeps the CSV clean -- we do not want pandas' row numbers.
    df.to_csv(out_path, index=False)
    return out_path


def load_derived(path: Path) -> pd.DataFrame:
    """Read a previously saved derived CSV back into a DataFrame."""
    return pd.read_csv(path)
