"""Tests for the SQLite build step (CREATE TABLE + INSERT + provenance)."""

from src.db import get_connection, read_table


def test_database_file_is_created(db_path):
    """Building the database should produce a file on disk."""
    assert db_path.exists()


def test_shoppers_table_has_all_rows(conn):
    """The shoppers table should hold every generated row (3 x 300)."""
    df = read_table(conn, "shoppers")
    assert len(df) == 900


def test_segment_has_three_values(conn):
    """The hidden ground-truth segment should have exactly three groups."""
    df = read_table(conn, "shoppers")
    assert set(df["segment"].unique()) == {0, 1, 2}


def test_basket_columns_are_binary(conn):
    """Each bought_* flag should round-trip as 0/1 integers."""
    df = read_table(conn, "shoppers")
    for col in [c for c in df.columns if c.startswith("bought_")]:
        assert set(df[col].unique()) <= {0, 1}


def test_provenance_table_records_the_source(conn):
    """A one-row provenance table should record where the data came from."""
    prov = read_table(conn, "provenance")
    assert len(prov) == 1
    assert prov.loc[0, "dataset"] == "shoppers"


def test_build_is_idempotent(db_path):
    """Re-running the build must not duplicate rows."""
    from db.build_sqlite import build

    build(db_path=db_path)
    conn = get_connection(db_path)
    try:
        shoppers = read_table(conn, "shoppers")
    finally:
        conn.close()
    assert len(shoppers) == 900
