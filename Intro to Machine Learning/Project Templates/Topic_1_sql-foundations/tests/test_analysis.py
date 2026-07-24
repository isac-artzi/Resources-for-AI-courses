"""Tests for the pandas analysis (merge/groupby) functions.

These use the ``raw_tables`` fixture, so they run without a database -- proof
that the analysis layer is pure pandas.
"""

import pytest

from src.analysis import (
    add_revenue_column,
    genre_popularity,
    revenue_by_employee,
    top_customers_by_spend,
)

# The revenue of the whole synthetic bookshop, summed by hand from invoice_items.
TOTAL_REVENUE = 38.16


def test_add_revenue_column(raw_tables):
    """revenue should equal unit_price times quantity, row by row."""
    items = add_revenue_column(raw_tables["invoice_items"])
    row = items.iloc[0]
    assert row["revenue"] == pytest.approx(row["unit_price"] * row["quantity"])


def test_top_customers_are_sorted_and_limited(raw_tables):
    """The result should be sorted high-to-low and respect top_n."""
    result = top_customers_by_spend(
        raw_tables["customers"],
        raw_tables["invoices"],
        raw_tables["invoice_items"],
        top_n=3,
    )
    assert len(result) == 3
    revenues = result["revenue"].tolist()
    assert revenues == sorted(revenues, reverse=True)


def test_genre_revenue_conserves_total(raw_tables):
    """Summing revenue across genres should recover the bookshop total."""
    result = genre_popularity(
        raw_tables["invoice_items"],
        raw_tables["books"],
        raw_tables["genres"],
    )
    assert result["revenue"].sum() == pytest.approx(TOTAL_REVENUE)


def test_employee_revenue_conserves_total(raw_tables):
    """Every dollar maps to exactly one employee, so the totals must match."""
    result = revenue_by_employee(
        raw_tables["employees"],
        raw_tables["customers"],
        raw_tables["invoices"],
        raw_tables["invoice_items"],
    )
    assert result["revenue"].sum() == pytest.approx(TOTAL_REVENUE)
