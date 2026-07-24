"""Page 1: Profile the raw data -- measure the mess before trusting it.

Profiling is the first thing a data scientist does with a new dataset. This
page shows the headline counts, a per-column quality report, and a numeric
summary, all computed by the pure functions in ``src/profile.py``.
"""

import streamlit as st

from src.profile import column_profile, numeric_summary, overview
from src.ui_data import load_table

st.title("1. Raw Profile")
st.write(
    "Before cleaning anything, we *measure* the data. Every number on this page "
    "comes from a small, tested function in `src/profile.py`."
)

# Load the raw signups table (built from data/raw/signups.csv on first run).
df = load_table("signups")

st.subheader("Overview")
stats = overview(df)
cols = st.columns(3)
cols[0].metric("Rows", stats["rows"])
cols[1].metric("Columns", stats["columns"])
cols[2].metric("Duplicate rows", stats["duplicate_rows"])
st.caption("Duplicate rows are exact copies -- they inflate every count and average.")

st.subheader("Per-column quality report")
st.write(
    "For each column: its type, how many values are missing, the missing "
    "percentage, and how many distinct values it has."
)
st.dataframe(column_profile(df), use_container_width=True)

st.subheader("Numeric summary")
st.write("Min / max / mean etc. for the numeric columns. Watch for impossible values.")
st.dataframe(numeric_summary(df), use_container_width=True)

st.warning(
    "Notice: `age` has a maximum of 999 (impossible), `income` has negative "
    "values, and both have missing values. The next pages fix all of this."
)
