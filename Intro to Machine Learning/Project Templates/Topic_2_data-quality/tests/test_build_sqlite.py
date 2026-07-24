"""Tests for the SQLite build step (CREATE TABLE + INSERT + provenance)."""

from src.db import get_connection, read_table


def test_database_file_is_created(db_path):
    """Building the database should produce a file on disk."""
    assert db_path.exists()


def test_signups_table_has_all_rows(conn):
    """The signups table should hold every raw row (including the duplicates)."""
    df = read_table(conn, "signups")
    # 400 base rows + 15 appended duplicates = 415.
    assert len(df) == 415


def test_provenance_table_records_the_source(conn):
    """A one-row provenance table should record where the data came from."""
    prov = read_table(conn, "provenance")
    assert len(prov) == 1
    assert prov.loc[0, "dataset"] == "signups"


def test_build_is_idempotent(db_path):
    """Re-running the build must not duplicate rows."""
    from db.build_sqlite import build

    build(db_path=db_path)
    conn = get_connection(db_path)
    try:
        signups = read_table(conn, "signups")
    finally:
        conn.close()
    assert len(signups) == 415
