"""Talk to the SQLite database.

Same role as earlier topics: SQLite is the raw-data store, used only for basic
CRUD on single tables. We read tables into pandas and do everything else
(vectorizing text, training the model) there.

This topic has a single raw table, ``emails``, plus a small ``provenance``
table that records where the data came from.
"""

import sqlite3
from pathlib import Path

import pandas as pd

from src.paths import DB_PATH

# The tables this project knows about. Validating table names against this set
# keeps the ``read_table`` helper safe.
KNOWN_TABLES = {"emails", "provenance"}


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open (and return) a connection to the SQLite database file."""
    return sqlite3.connect(db_path)


def read_query(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> pd.DataFrame:
    """Run a SELECT and return the rows as a pandas DataFrame.

    Use ``?`` placeholders for any user-supplied values -- never build SQL by
    gluing strings together, which invites SQL-injection bugs.
    """
    return pd.read_sql(sql, conn, params=params)


def read_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    """Read an entire known table into its own DataFrame."""
    if table not in KNOWN_TABLES:
        raise ValueError(f"Unknown table {table!r}; expected one of {sorted(KNOWN_TABLES)}")
    return read_query(conn, f"SELECT * FROM {table}")
