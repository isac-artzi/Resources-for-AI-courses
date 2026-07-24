"""Reusable pandas cleaning helpers.

Real data is messy: stray whitespace, inconsistent capitalization, duplicate
rows, dates stored as text. This module collects small, single-purpose
cleaning functions. Each one:
    * does exactly one thing,
    * has a docstring explaining what and why,
    * returns a new DataFrame (it never edits the input in place),
    * has a matching unit test in tests/test_clean.py.

The synthetic sample data is already tidy, so these functions mostly leave it
unchanged -- but they show the patterns you will need on real datasets.
"""

import pandas as pd


def standardize_country(customers: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and normalize the capitalization of the country column.

    For example ' usa ' and 'USA' should be treated as the same country.
    We upper-case here because the sample data uses codes like 'USA'.
    """
    result = customers.copy()
    result["country"] = result["country"].str.strip().str.upper()
    return result


def drop_duplicate_customers(customers: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate customer rows, keeping the first occurrence.

    Duplicates inflate counts and revenue, so removing them early prevents
    misleading results downstream.
    """
    return customers.drop_duplicates().reset_index(drop=True)


def parse_invoice_dates(invoices: pd.DataFrame) -> pd.DataFrame:
    """Convert the text ``invoice_date`` column into real datetime values.

    SQLite stored the dates as plain text. Parsing them into datetimes lets us
    sort chronologically and extract the month or year later.
    """
    result = invoices.copy()
    result["invoice_date"] = pd.to_datetime(result["invoice_date"], format="%Y-%m-%d")
    return result
