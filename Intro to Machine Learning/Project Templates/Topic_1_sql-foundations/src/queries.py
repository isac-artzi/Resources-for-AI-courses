"""The six basic single-table SELECT queries.

Every query here touches EXACTLY ONE table and uses only introductory SQL:
SELECT with column projection, WHERE, ORDER BY, LIMIT, and one simple
single-column GROUP BY with COUNT(*). There are deliberately NO JOINs,
subqueries, HAVING clauses, or window functions -- those are out of scope for
this course, and all multi-table work happens in pandas (see analysis.py).

Each query is stored as a plain string so the Streamlit UI can display the SQL
text next to its result.
"""

import sqlite3

import pandas as pd

from src.db import read_query

# A dictionary of {short name: SQL text}. Keeping them here (not scattered
# through the UI) makes them easy to read, test, and show on screen.
QUERIES: dict[str, str] = {
    # 1. Projection + WHERE + ORDER BY: the priciest books first.
    "premium_books": (
        "SELECT title, unit_price "
        "FROM books "
        "WHERE unit_price > 0.99 "
        "ORDER BY unit_price DESC, title ASC"
    ),
    # 2. WHERE only: customers located in the USA.
    "usa_customers": (
        "SELECT customer_id, first_name, last_name, country "
        "FROM customers "
        "WHERE country = 'USA'"
    ),
    # 3. ORDER BY + LIMIT: the first five book titles alphabetically.
    "first_five_books": ("SELECT title FROM books ORDER BY title ASC LIMIT 5"),
    # 4. Single-column GROUP BY with COUNT(*): how many books per genre.
    "books_per_genre": (
        "SELECT genre_id, COUNT(*) AS book_count "
        "FROM books "
        "GROUP BY genre_id "
        "ORDER BY book_count DESC"
    ),
    # 5. Single-column GROUP BY with COUNT(*): how many customers per country.
    "customers_per_country": (
        "SELECT country, COUNT(*) AS customer_count "
        "FROM customers "
        "GROUP BY country "
        "ORDER BY customer_count DESC"
    ),
    # 6. Projection + ORDER BY: series titles in alphabetical order.
    "series_alphabetical": ("SELECT title FROM series ORDER BY title ASC"),
}


def run_query(conn: sqlite3.Connection, name: str) -> pd.DataFrame:
    """Run one of the named queries and return its result as a DataFrame."""
    if name not in QUERIES:
        raise KeyError(f"Unknown query {name!r}; choose from {sorted(QUERIES)}")
    return read_query(conn, QUERIES[name])
