"""Demonstrate the last two CRUD operations: UPDATE and DELETE.

``build_sqlite.py`` covers CREATE and INSERT. This companion script shows the
other two basic single-table operations you are expected to know:

    * UPDATE -- change existing rows (here: fix a typo in an author name).
    * DELETE -- remove rows (here: drop a leftover test book).

Run it after building the database:

    python db/build_sqlite.py
    python db/crud_maintenance.py
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import DB_PATH  # noqa: E402  (import after sys.path fix)


def fix_author_typo(conn: sqlite3.Connection) -> int:
    """UPDATE example: correct an author's name.

    Imagine 'Silas Wren' was entered as 'Silas Wrenn'. This is exactly the
    kind of small correction an UPDATE is for. Returns the number of rows
    changed so a test can confirm it worked.
    """
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE authors SET name = ? WHERE name = ?",
        ("Silas Wren", "Silas Wrenn"),
    )
    conn.commit()
    return cursor.rowcount


def delete_test_book(conn: sqlite3.Connection, title: str = "TEST TITLE") -> int:
    """DELETE example: remove any leftover test/placeholder book.

    Returns the number of rows deleted (0 if there was nothing to remove).
    """
    cursor = conn.cursor()
    cursor.execute("DELETE FROM books WHERE title = ?", (title,))
    conn.commit()
    return cursor.rowcount


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    try:
        updated = fix_author_typo(conn)
        deleted = delete_test_book(conn)
        print(f"UPDATE changed {updated} row(s); DELETE removed {deleted} row(s).")
    finally:
        conn.close()
