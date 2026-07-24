"""All multi-table analysis -- done in pandas, not SQL.

This is the heart of the course's build pattern. We pulled each table out of
SQLite as its own DataFrame; now we combine them with ``DataFrame.merge``.

``DataFrame.merge`` is the pandas equivalent of a SQL JOIN: it lines up rows
from two tables using a shared key column (for example ``genre_id``). We do the
joining here, in Python, because it is easier to read, test, and debug than
multi-table SQL -- and because this course keeps SQL restricted to basic,
single-table CRUD.

Every function takes plain DataFrames and returns a plain DataFrame, which
makes them easy to unit-test without a database.
"""

import pandas as pd


def add_revenue_column(invoice_items: pd.DataFrame) -> pd.DataFrame:
    """Add a ``revenue`` column = unit_price x quantity to each line item.

    Returns a COPY so the caller's DataFrame is not modified in place.
    """
    result = invoice_items.copy()
    result["revenue"] = result["unit_price"] * result["quantity"]
    return result


def top_customers_by_spend(
    customers: pd.DataFrame,
    invoices: pd.DataFrame,
    invoice_items: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return the highest-spending customers.

    Data flow (each ``merge`` is a JOIN done in pandas):
        invoice_items (+ revenue)
          --merge on invoice_id-->  invoices     (to learn the customer_id)
          --merge on customer_id--> customers    (to learn the customer's name)
        then group by customer and sum the revenue.
    """
    items = add_revenue_column(invoice_items)

    # Attach the customer_id that each line item belongs to.
    items = items.merge(invoices[["invoice_id", "customer_id"]], on="invoice_id", how="inner")

    # Attach each customer's name and country.
    items = items.merge(
        customers[["customer_id", "first_name", "last_name", "country"]],
        on="customer_id",
        how="inner",
    )

    # Sum revenue per customer. ``groupby`` here plays the role of SQL GROUP BY.
    grouped = (
        items.groupby(["customer_id", "first_name", "last_name", "country"], as_index=False)[
            "revenue"
        ]
        .sum()
        .sort_values("revenue", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return grouped


def genre_popularity(
    invoice_items: pd.DataFrame,
    books: pd.DataFrame,
    genres: pd.DataFrame,
) -> pd.DataFrame:
    """Return total revenue and units sold per genre.

    Data flow:
        invoice_items --merge on book_id--> books  (to learn each book's genre)
                       --merge on genre_id--> genres  (to learn the genre name)
        then group by genre name.
    """
    items = add_revenue_column(invoice_items)

    items = items.merge(books[["book_id", "genre_id"]], on="book_id", how="inner")
    items = items.merge(genres[["genre_id", "name"]], on="genre_id", how="inner")

    grouped = (
        items.groupby("name", as_index=False)
        .agg(revenue=("revenue", "sum"), units_sold=("quantity", "sum"))
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    # Rename the generic ``name`` column to something clearer for the UI.
    return grouped.rename(columns={"name": "genre"})


def revenue_by_employee(
    employees: pd.DataFrame,
    customers: pd.DataFrame,
    invoices: pd.DataFrame,
    invoice_items: pd.DataFrame,
) -> pd.DataFrame:
    """Return total revenue attributed to each sales-rep employee.

    Data flow:
        invoice_items --> invoices   (customer_id)
                      --> customers   (support_rep_id)
                      --> employees   (employee name; support_rep_id == employee_id)
    """
    items = add_revenue_column(invoice_items)

    items = items.merge(invoices[["invoice_id", "customer_id"]], on="invoice_id", how="inner")
    items = items.merge(
        customers[["customer_id", "support_rep_id"]], on="customer_id", how="inner"
    )
    # The customer's ``support_rep_id`` matches an ``employee_id``. We tell
    # merge to line those two differently named columns up with left_on/right_on.
    items = items.merge(
        employees[["employee_id", "first_name", "last_name"]],
        left_on="support_rep_id",
        right_on="employee_id",
        how="inner",
    )

    grouped = (
        items.groupby(["employee_id", "first_name", "last_name"], as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    return grouped
