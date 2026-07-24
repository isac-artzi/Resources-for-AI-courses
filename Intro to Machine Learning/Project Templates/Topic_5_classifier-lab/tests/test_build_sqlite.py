"""Tests for the SQLite build step (CREATE TABLE + INSERT + provenance)."""

from src.db import get_connection, read_table


def test_database_file_is_created(db_path):
    """Building the database should produce a file on disk."""
    assert db_path.exists()


def test_customers_table_has_all_rows(conn):
    """The customers table should hold every generated row."""
    df = read_table(conn, "customers")
    assert len(df) == 2000


def test_target_is_binary(conn):
    """The churned label should round-trip as 0/1 integers."""
    df = read_table(conn, "customers")
    assert set(df["churned"].unique()) == {0, 1}


def test_provenance_table_records_the_source(conn):
    """A one-row provenance table should record where the data came from."""
    prov = read_table(conn, "provenance")
    assert len(prov) == 1
    assert prov.loc[0, "dataset"] == "customers"


def test_build_is_idempotent(db_path):
    """Re-running the build must not duplicate rows."""
    from db.build_sqlite import build

    build(db_path=db_path)
    conn = get_connection(db_path)
    try:
        customers = read_table(conn, "customers")
    finally:
        conn.close()
    assert len(customers) == 2000
