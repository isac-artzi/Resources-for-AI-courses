"""Talk to the SQLite database.

In this course SQLite is used ONLY as a simple raw-data store, with *basic
CRUD on single tables* -- CREATE TABLE, INSERT, single-table SELECT, UPDATE,
DELETE. We deliberately do NOT use JOINs, subqueries, or window functions in
SQL. All joining and heavy analysis happens later, in pandas (see
``src/analysis.py``).

This module gives you two small, reusable helpers:
    * ``get_connection`` -- open a connection to the SQLite file.
    * ``read_query``     -- run a SELECT and get the result back as a pandas
                            DataFrame (the bridge from SQL to pandas).
"""

import sqlite3
from pathlib import Path

import pandas as pd

from src.paths import DB_PATH


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open (and return) a connection to the SQLite database file.

    Parameters
    ----------
    db_path:
        Location of the ``.sqlite`` file. Defaults to the project database
        built by ``db/build_sqlite.py``.

    Notes
    -----
    The caller is responsible for closing the connection (or using it inside
    a ``with`` block). SQLite connections are cheap to open.
    """
    return sqlite3.connect(db_path)


def read_query(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> pd.DataFrame:
    """Run a SELECT statement and return the rows as a pandas DataFrame.

    ``pandas.read_sql`` is the standard bridge from a SQL result set into a
    DataFrame. Once the data is in a DataFrame we can clean, join, and
    aggregate it with pandas.

    Parameters
    ----------
    conn:
        An open SQLite connection (from :func:`get_connection`).
    sql:
        A single-table SELECT statement.
    params:
        Optional values for ``?`` placeholders in ``sql``. Using placeholders
        (instead of putting values directly into the string) is the safe way
        to pass user input -- it prevents SQL-injection bugs.
    """
    return pd.read_sql(sql, conn, params=params)


def read_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    """Read an entire table into its own DataFrame.

    This is the most common first step in the build pattern: pull each table
    out of SQLite so pandas can take over. Table names cannot be passed as
    ``?`` parameters, so we validate the name against the known tables to stay
    safe.
    """
    allowed = {
        "genres",
        "authors",
        "series",
        "books",
        "employees",
        "customers",
        "invoices",
        "invoice_items",
    }
    if table not in allowed:
        raise ValueError(f"Unknown table {table!r}; expected one of {sorted(allowed)}")
    # The table name is now known-safe, so it is fine to format it in.
    return read_query(conn, f"SELECT * FROM {table}")
