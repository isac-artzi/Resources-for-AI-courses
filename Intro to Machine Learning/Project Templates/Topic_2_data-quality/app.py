"""Home page of the Data-Quality Dashboard product.

This is the file Streamlit runs first (``streamlit run app.py``). It introduces
the product and explains the build pattern. The individual steps live in the
``pages/`` folder and appear automatically in the sidebar.

The whole app follows one data flow, the universal build pattern:

    raw CSV  ->  SQLite (basic CRUD)  ->  pandas (profile/clean)  ->  derived CSV  ->  this UI
"""

import streamlit as st

from src.profile import overview
from src.ui_data import load_table

# ``set_page_config`` must be the first Streamlit call. It sets the browser tab
# title and uses the wider layout so tables have room to breathe.
st.set_page_config(page_title="Data-Quality Dashboard", page_icon="🧹", layout="wide")

st.title("🧹 Data-Quality Dashboard")
st.caption("An introductory data-profiling + cleaning + validation product template.")

st.markdown(
    """
Real-world data is messy: missing values, duplicates, impossible numbers, and
inconsistent text. This app takes a deliberately messy `signups` dataset and
walks through the three habits of good data hygiene:

- **Raw Profile** — measure the mess: rows, types, missing values, duplicates.
- **Schema Report** — write down what *good* data looks like and check against it.
- **Cleaned Explorer** — run a re-runnable cleaning pipeline and confirm it passes.

### The build pattern this project follows
"""
)

# A tiny visual reminder of the pipeline every page relies on.
st.code(
    "data/raw/signups.csv  ->  db/build_sqlite.py  ->  SQLite (CRUD)\n"
    "SQLite tables         ->  pandas.read_sql     ->  DataFrame\n"
    "DataFrame             ->  src/profile.py      ->  quality report\n"
    "DataFrame             ->  src/clean.py        ->  cleaned DataFrame\n"
    "cleaned data          ->  src/schema.py       ->  pandera validation\n"
    "cleaned data          ->  src/persist.py      ->  data/derived/*.csv",
    language="text",
)

# Load the raw table once (cached). On the very first run this also builds the
# SQLite database from the raw CSV.
signups = load_table("signups")
stats = overview(signups)

st.subheader("Raw dataset at a glance")
cols = st.columns(3)
cols[0].metric("Rows", stats["rows"])
cols[1].metric("Columns", stats["columns"])
cols[2].metric("Duplicate rows", stats["duplicate_rows"])

st.info(
    "Start with **Raw Profile** to see the problems, then read the "
    "**Schema Report** (the raw data fails it), then open **Cleaned Explorer** "
    "to fix the data and watch it pass."
)
