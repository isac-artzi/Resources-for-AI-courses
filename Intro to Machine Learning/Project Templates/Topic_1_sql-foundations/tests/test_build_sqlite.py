"""Tests for the SQLite build step (CREATE TABLE + INSERT)."""

from src.db import get_connection, read_table


def test_database_file_is_created(db_path):
    """Building the database should produce a file on disk."""
    assert db_path.exists()


def test_all_tables_have_expected_row_counts(conn):
    """Each table should contain exactly the rows from its CSV."""
    expected = {
        "genres": 5,
        "authors": 4,
        "series": 6,
        "books": 15,
        "employees": 3,
        "customers": 8,
        "invoices": 12,
        "invoice_items": 24,
    }
    for table, count in expected.items():
        df = read_table(conn, table)
        assert len(df) == count, f"{table} had {len(df)} rows, expected {count}"


def test_build_is_idempotent(db_path):
    """Re-running the build must not duplicate rows."""
    # db_path fixture already built once; build again over the same file.
    from db.build_sqlite import build

    build(db_path=db_path)
    conn = get_connection(db_path)
    try:
        books = read_table(conn, "books")
    finally:
        conn.close()
    assert len(books) == 15


def test_book_columns_are_typed(conn):
    """The price column should come back as a number, not text."""
    books = read_table(conn, "books")
    assert books["unit_price"].dtype.kind == "f"  # 'f' == float
