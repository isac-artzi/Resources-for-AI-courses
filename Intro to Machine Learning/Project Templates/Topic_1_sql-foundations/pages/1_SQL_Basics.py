"""Page 1 — SQL Basics.

Shows each of the six single-table SELECT queries: the exact SQL text and the
rows it returns. This is the "SQL read" half of the build pattern. Notice that
none of these queries JOIN tables -- that comes later, in pandas.
"""

import streamlit as st

from src.db import get_connection
from src.queries import QUERIES, run_query
from src.ui_data import ensure_database

st.title("🗃️ SQL Basics")
st.write(
    "Six basic, single-table queries. Each uses only introductory SQL: "
    "`SELECT`, `WHERE`, `ORDER BY`, `LIMIT`, and a simple `GROUP BY ... COUNT(*)`."
)

# Short, human-friendly explanation for each query key.
DESCRIPTIONS = {
    "premium_books": "Projection + WHERE + ORDER BY: books priced above $0.99.",
    "usa_customers": "WHERE: customers located in the USA.",
    "first_five_books": "ORDER BY + LIMIT: the first five book titles.",
    "books_per_genre": "GROUP BY + COUNT(*): number of books in each genre.",
    "customers_per_country": "GROUP BY + COUNT(*): number of customers per country.",
    "series_alphabetical": "Projection + ORDER BY: series titles A–Z.",
}

# Make sure the database exists, then open one connection for all six queries.
ensure_database()
conn = get_connection()
try:
    for name, sql in QUERIES.items():
        st.subheader(name.replace("_", " ").title())
        st.caption(DESCRIPTIONS[name])
        # Show the SQL the way a code reviewer would want to see it.
        st.code(sql, language="sql")
        # Run it and show the resulting rows.
        result = run_query(conn, name)
        st.dataframe(result, use_container_width=True)
finally:
    conn.close()
