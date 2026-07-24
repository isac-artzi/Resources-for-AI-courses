"""Tests for the SQLite build step (CREATE TABLE + INSERT + provenance)."""

from src.db import get_connection, read_table


def test_database_file_is_created(db_path):
    """Building the database should produce a file on disk."""
    assert db_path.exists()


def test_emails_table_has_all_rows(conn):
    """The emails table should hold every message (400 spam + 400 ham)."""
    df = read_table(conn, "emails")
    assert len(df) == 800


def test_both_labels_present(conn):
    """Both classes should survive the round-trip through SQLite."""
    df = read_table(conn, "emails")
    assert set(df["label"].unique()) == {"spam", "ham"}


def test_provenance_table_records_the_source(conn):
    """A one-row provenance table should record where the data came from."""
    prov = read_table(conn, "provenance")
    assert len(prov) == 1
    assert prov.loc[0, "dataset"] == "emails"


def test_build_is_idempotent(db_path):
    """Re-running the build must not duplicate rows."""
    from db.build_sqlite import build

    build(db_path=db_path)
    conn = get_connection(db_path)
    try:
        emails = read_table(conn, "emails")
    finally:
        conn.close()
    assert len(emails) == 800
