"""Page 3: Clean the data, confirm it passes, and save the result.

This page runs the cleaning pipeline from ``src/clean.py``, re-profiles the
result, re-validates it against the same schema (now it passes), and lets you
save the cleaned data as a dated derived CSV.
"""

import streamlit as st

from src.clean import run_cleaning
from src.persist import save_derived
from src.profile import column_profile, overview
from src.schema import validate
from src.ui_data import load_table

st.title("3. Cleaned Explorer")
st.write(
    "Now we fix the data with the re-runnable pipeline in `src/clean.py`. Each "
    "step does one thing: drop duplicates, fix types, standardize text, mark "
    "impossible values missing, then fill the gaps with the median."
)

raw = load_table("signups")
cleaned = run_cleaning(raw)

st.subheader("Before vs after")
before, after = overview(raw), overview(cleaned)
dup_delta = after["duplicate_rows"] - before["duplicate_rows"]
cols = st.columns(3)
cols[0].metric("Rows", after["rows"], delta=after["rows"] - before["rows"])
cols[1].metric("Duplicate rows", after["duplicate_rows"], delta=dup_delta)
cols[2].metric("Columns", after["columns"])

st.subheader("Cleaned per-column report")
# After cleaning there should be no missing values left.
st.dataframe(column_profile(cleaned), use_container_width=True)

st.subheader("Re-validating the CLEANED data")
ok, failures = validate(cleaned)
if ok:
    st.success("The cleaned data passes every rule in the schema. 🎉")
else:
    st.error("Unexpected: cleaning did not satisfy the schema.")
    st.dataframe(failures, use_container_width=True)

st.subheader("Preview")
st.dataframe(cleaned.head(20), use_container_width=True)

st.subheader("Save the cleaned data")
st.write(
    "Streamlit Cloud's file system is temporary, so we persist cleaned results "
    "as a dated CSV under `data/derived/` -- our convention for derived state."
)
if st.button("Save cleaned signups to a dated CSV"):
    path = save_derived(cleaned, "signups")
    st.success(f"Saved {path.name} ({len(cleaned)} rows).")
