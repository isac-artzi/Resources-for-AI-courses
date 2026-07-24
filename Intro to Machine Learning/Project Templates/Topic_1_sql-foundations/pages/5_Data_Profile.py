"""Page 5 — Data Profile.

Before trusting any analysis, profile the data: how many rows, what types, and
where are the missing values? Render this profile in the app itself (not pasted
into the README), one table at a time.
"""

import pandas as pd
import streamlit as st

from src.ui_data import load_all_tables

st.title("🔎 Data Profile")
st.write(
    "A quick health check of every table: shape, column types, null counts, "
    "and basic statistics. Run this first on any new dataset."
)

tables = load_all_tables()

# Let the user pick which table to inspect.
table_name = st.selectbox("Choose a table to profile", list(tables.keys()))
df = tables[table_name]

# Headline numbers.
c1, c2 = st.columns(2)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])

# Per-column profile: dtype and how many values are missing.
st.subheader("Columns")
profile = pd.DataFrame(
    {
        "dtype": df.dtypes.astype(str),
        "null_count": df.isnull().sum(),
        "distinct_values": df.nunique(),
    }
)
st.dataframe(profile, use_container_width=True)

# Summary statistics for numeric columns (min/max/mean/etc.).
numeric = df.select_dtypes(include="number")
if not numeric.empty:
    st.subheader("Numeric summary")
    st.dataframe(numeric.describe(), use_container_width=True)
else:
    st.caption("This table has no numeric columns to summarize.")

st.subheader("Preview")
st.dataframe(df.head(), use_container_width=True)
