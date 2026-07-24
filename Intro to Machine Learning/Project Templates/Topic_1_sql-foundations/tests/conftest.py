"""Shared pytest fixtures.

A *fixture* is reusable setup code. These build a fresh SQLite database in a
temporary folder for each test that needs one, so tests never depend on each
other and never touch your real ``db/bookshop.sqlite``.
"""

import sqlite3

import pandas as pd
import pytest

from db.build_sqlite import TABLES, build
from src.paths import RAW_DIR


@pytest.fixture()
def db_path(tmp_path):
    """Build a throwaway database in pytest's temporary directory."""
    path = tmp_path / "test.sqlite"
    build(db_path=path, raw_dir=RAW_DIR)
    return path


@pytest.fixture()
def conn(db_path):
    """Open a connection to the throwaway database and close it afterwards."""
    connection = sqlite3.connect(db_path)
    yield connection
    connection.close()


@pytest.fixture()
def raw_tables():
    """Load the raw CSVs straight into DataFrames (no database needed).

    Handy for testing the pandas analysis functions in isolation.
    """
    names = [name.removesuffix(".csv") for name in TABLES.values()]
    return {name: pd.read_csv(RAW_DIR / f"{name}.csv") for name in names}
