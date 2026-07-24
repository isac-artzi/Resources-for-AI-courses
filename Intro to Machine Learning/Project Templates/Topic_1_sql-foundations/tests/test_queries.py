"""Tests for the six single-table SELECT queries."""

from src.queries import QUERIES, run_query


def test_all_queries_return_rows(conn):
    """Every named query should run and return at least one row."""
    for name in QUERIES:
        result = run_query(conn, name)
        assert len(result) > 0, f"query {name} returned no rows"


def test_premium_books_are_all_above_threshold(conn):
    """The WHERE clause should exclude any book priced at or below $0.99."""
    result = run_query(conn, "premium_books")
    assert (result["unit_price"] > 0.99).all()


def test_first_five_books_respects_limit(conn):
    """LIMIT 5 should return exactly five rows."""
    result = run_query(conn, "first_five_books")
    assert len(result) == 5


def test_books_per_genre_totals_all_books(conn):
    """The per-genre counts should add up to the 15 total books."""
    result = run_query(conn, "books_per_genre")
    assert result["book_count"].sum() == 15


def test_customers_per_country_totals_all_customers(conn):
    """The per-country counts should add up to the 8 total customers."""
    result = run_query(conn, "customers_per_country")
    assert result["customer_count"].sum() == 8
