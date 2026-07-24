"""Home page of the Bookshop Analytics product.

This is the file Streamlit runs first (``streamlit run app.py``). It introduces
the product and explains the build pattern. The individual analyses live in the
``pages/`` folder and appear automatically in the sidebar.

The whole app follows one data flow, the universal build pattern:

    raw CSVs  ->  SQLite (basic CRUD)  ->  pandas (merge/clean)  ->  derived CSV  ->  this UI
"""

import streamlit as st

from src.ui_data import load_all_tables

# ``set_page_config`` must be the first Streamlit call. It sets the browser tab
# title and uses the wider layout so tables have room to breathe.
st.set_page_config(page_title="Bookshop Analytics", page_icon="📚", layout="wide")

st.title("📚 Bookshop Analytics")
st.caption("An introductory SQL + pandas + Streamlit product template.")

st.markdown(
    """
This app answers business questions about a small online bookshop. Use the
pages in the sidebar to explore the data:

- **SQL Basics** — the six single-table SELECT queries that read the raw data.
- **Top Customers** — who spends the most (joined with `pandas.merge`).
- **Genre Popularity** — which genres earn the most revenue.
- **Revenue by Employee** — which sales rep is tied to which revenue.
- **Data Profile** — shape, types, and missing-value checks for every table.

### The build pattern this project follows
"""
)

# A tiny visual reminder of the pipeline every page relies on.
st.code(
    "data/raw/*.csv  ->  db/build_sqlite.py  ->  SQLite (CRUD)\n"
    "SQLite tables   ->  pandas.read_sql     ->  one DataFrame per table\n"
    "DataFrames      ->  DataFrame.merge      ->  joined analysis (pandas)\n"
    "analysis        ->  src/persist.py       ->  data/derived/*.csv\n"
    "derived CSV     ->  Streamlit            ->  the pages you see here",
    language="text",
)

# Load every table once (cached). On the very first run this also builds the
# SQLite database from the raw CSVs.
tables = load_all_tables()

st.subheader("Dataset at a glance")
# Show one small metric per table so a new user sees the shape of the data.
cols = st.columns(4)
for i, (name, df) in enumerate(tables.items()):
    cols[i % 4].metric(label=name, value=f"{len(df)} rows")

st.info(
    "Tip: every analysis page shows both the **SQL read** (basic, single-table) "
    "and the **pandas analysis** (all joining and aggregation) so you can see "
    "exactly how the data flows."
)
