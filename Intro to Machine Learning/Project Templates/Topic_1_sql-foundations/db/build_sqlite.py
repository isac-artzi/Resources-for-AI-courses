"""Build the SQLite database from the raw CSV files using basic CRUD.

This script demonstrates the FIRST step of the universal build pattern:
    raw CSV  -->  SQLite (via CREATE TABLE + INSERT)

We load ONE table at a time. We never JOIN here -- joining is a pandas job.
The script is *idempotent*: running it again drops and rebuilds each table,
so you always get a clean, predictable database. It runs both on your laptop
and during Streamlit Cloud's build phase.

Run it directly:

    python db/build_sqlite.py
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

# When this file is run directly (``python db/build_sqlite.py``) the repo root
# is not automatically on the import path, so ``import src`` would fail. We add
# the repo root (the parent of this file's ``db/`` folder) so the import works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import DB_PATH, RAW_DIR  # noqa: E402  (import after sys.path fix)

# Which CSV file becomes which SQLite table. The order does not matter because
# we do not use foreign-key constraints at this introductory level.
TABLES = {
    "genres": "genres.csv",
    "authors": "authors.csv",
    "series": "series.csv",
    "books": "books.csv",
    "employees": "employees.csv",
    "customers": "customers.csv",
    "invoices": "invoices.csv",
    "invoice_items": "invoice_items.csv",
}


def _sqlite_type(dtype) -> str:
    """Map a pandas column dtype to a SQLite column type.

    SQLite has very few types. We only need three:
        * INTEGER for whole numbers,
        * REAL for decimals (like a price),
        * TEXT for everything else (names, dates as strings).
    """
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(dtype):
        return "REAL"
    return "TEXT"


def create_and_fill_table(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> None:
    """Create one table and INSERT all of its rows (basic CRUD).

    Steps, all on a SINGLE table:
        1. DROP TABLE IF EXISTS  -- makes the script safe to re-run.
        2. CREATE TABLE          -- one column per CSV column.
        3. INSERT INTO           -- one row per CSV row.
    """
    cursor = conn.cursor()

    # 1. Start clean so re-running gives identical results (idempotent).
    cursor.execute(f"DROP TABLE IF EXISTS {table}")

    # 2. Build a CREATE TABLE statement from the DataFrame's columns.
    #    Example result: CREATE TABLE genres (genre_id INTEGER, name TEXT)
    column_defs = ", ".join(f"{col} {_sqlite_type(df[col].dtype)}" for col in df.columns)
    cursor.execute(f"CREATE TABLE {table} ({column_defs})")

    # 3. INSERT every row. The "?" marks are placeholders that SQLite fills in
    #    safely; ``executemany`` inserts the whole list of rows in one call.
    placeholders = ", ".join("?" for _ in df.columns)
    column_names = ", ".join(df.columns)
    rows = df.itertuples(index=False, name=None)  # each row as a plain tuple
    cursor.executemany(
        f"INSERT INTO {table} ({column_names}) VALUES ({placeholders})",
        rows,
    )

    conn.commit()


def build(db_path: Path = DB_PATH, raw_dir: Path = RAW_DIR) -> Path:
    """Build the whole database and return the path to the created file."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        for table, csv_name in TABLES.items():
            df = pd.read_csv(raw_dir / csv_name)
            create_and_fill_table(conn, table, df)
            print(f"  built table {table:<14} ({len(df)} rows)")
    finally:
        conn.close()
    return db_path


if __name__ == "__main__":
    print(f"Building SQLite database at {DB_PATH} ...")
    build()
    print("Done.")
