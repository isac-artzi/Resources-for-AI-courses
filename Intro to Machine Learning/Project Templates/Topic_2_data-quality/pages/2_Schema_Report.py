"""Page 2: Validate against a schema -- the raw data should FAIL.

A schema is a written contract for what clean data looks like. This page runs
the raw data through it. Because the raw data is messy, validation fails, and
we show exactly which rules were broken and by how many rows.
"""

import streamlit as st

from src.schema import SIGNUPS_SCHEMA, validate
from src.ui_data import load_table

st.title("2. Schema Report")
st.write(
    "A **schema** writes down the rules for good data: each column's type and "
    "the values we accept. `src/schema.py` turns those rules into code with the "
    "`pandera` library."
)

st.subheader("The contract")
# Show the schema so students can read the rules directly.
st.code(str(SIGNUPS_SCHEMA), language="text")

df = load_table("signups")

st.subheader("Validating the RAW data")
ok, failures = validate(df)

if ok:
    st.success("The data passes every rule.")
else:
    st.error(
        "The raw data FAILS validation -- exactly as expected. Each row below is "
        "one broken rule (which column, which check, and the offending value)."
    )
    # ``failure_cases`` is a DataFrame; show the most common failures first.
    st.dataframe(failures, use_container_width=True)
    st.metric("Total rule violations", len(failures))

st.info(
    "This is the 'before' picture. Open **Cleaned Explorer** to run the cleaning "
    "pipeline and see the same schema pass."
)
