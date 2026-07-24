"""Tests for the SQLite build step (CREATE TABLE + INSERT + provenance)."""

from src.db import get_connection, read_table


def test_database_file_is_created(db_path):
    """Building the database should produce a file on disk."""
    assert db_path.exists()


def test_prices_table_has_all_rows(conn):
    """The prices table should hold every raw trading day."""
    df = read_table(conn, "prices")
    assert len(df) == 500


def test_close_is_numeric(conn):
    """The close price should come back as a number, not text."""
    df = read_table(conn, "prices")
    assert df["close"].dtype.kind == "f"


def test_provenance_table_records_the_source(conn):
    """A one-row provenance table should record where the data came from."""
    prov = read_table(conn, "provenance")
    assert len(prov) == 1
    assert prov.loc[0, "dataset"] == "prices"


def test_build_is_idempotent(db_path):
    """Re-running the build must not duplicate rows."""
    from db.build_sqlite import build

    build(db_path=db_path)
    conn = get_connection(db_path)
    try:
        prices = read_table(conn, "prices")
    finally:
        conn.close()
    assert len(prices) == 500
