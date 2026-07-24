"""Tests for the UPDATE and DELETE maintenance operations."""

from db.crud_maintenance import delete_test_book, fix_author_typo
from src.db import read_query


def test_update_fixes_a_typo(conn):
    """After introducing a typo, the UPDATE should correct exactly one row."""
    # Introduce a typo so there is something for the UPDATE to fix.
    conn.execute("UPDATE authors SET name = 'Silas Wrenn' WHERE name = 'Silas Wren'")
    conn.commit()

    changed = fix_author_typo(conn)
    assert changed == 1

    # The corrected name should now be present again.
    names = read_query(conn, "SELECT name FROM authors")["name"].tolist()
    assert "Silas Wren" in names
    assert "Silas Wrenn" not in names


def test_delete_removes_only_matching_rows(conn):
    """DELETE should remove a planted test row and leave real data intact."""
    conn.execute(
        "INSERT INTO books (book_id, title, series_id, genre_id, unit_price) "
        "VALUES (999, 'TEST TITLE', 1, 1, 0.0)"
    )
    conn.commit()

    removed = delete_test_book(conn)
    assert removed == 1

    remaining = read_query(conn, "SELECT COUNT(*) AS n FROM books")["n"].iloc[0]
    assert remaining == 15  # back to the original count


def test_delete_is_safe_when_nothing_matches(conn):
    """Deleting a non-existent test book should remove zero rows."""
    removed = delete_test_book(conn)
    assert removed == 0
