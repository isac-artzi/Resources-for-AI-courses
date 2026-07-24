"""Tests for the SQLite build step (CREATE TABLE + INSERT + provenance)."""

from src.db import get_connection, read_table


def test_database_file_is_created(db_path):
    """Building the database should produce a file on disk."""
    assert db_path.exists()


def test_houses_table_has_all_rows(conn):
    """The houses table should hold every generated row."""
    df = read_table(conn, "houses")
    assert len(df) == 1500


def test_price_is_numeric(conn):
    """The price target should come back as a number, not text."""
    df = read_table(conn, "houses")
    assert df["price"].dtype.kind in "if"  # int or float


def test_provenance_table_records_the_source(conn):
    """A one-row provenance table should record where the data came from."""
    prov = read_table(conn, "provenance")
    assert len(prov) == 1
    assert prov.loc[0, "dataset"] == "houses"


def test_build_is_idempotent(db_path):
    """Re-running the build must not duplicate rows."""
    from db.build_sqlite import build

    build(db_path=db_path)
    conn = get_connection(db_path)
    try:
        houses = read_table(conn, "houses")
    finally:
        conn.close()
    assert len(houses) == 1500
