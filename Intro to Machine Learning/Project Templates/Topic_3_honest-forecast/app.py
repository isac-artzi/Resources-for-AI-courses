"""Home page of the Honest Forecast product.

This is the file Streamlit runs first (``streamlit run app.py``). It introduces
the product and explains the build pattern. The individual steps live in the
``pages/`` folder and appear automatically in the sidebar.

The whole app follows one data flow, the universal build pattern:

    raw CSV  ->  SQLite (basic CRUD)  ->  pandas + sklearn  ->  derived CSV  ->  Streamlit UI
"""

import streamlit as st

from src.ui_data import load_table

# ``set_page_config`` must be the first Streamlit call.
st.set_page_config(page_title="Honest Forecast", page_icon="📈", layout="wide")

st.title("📈 Honest Forecast")
st.caption("An introductory linear-regression time-series product template.")

st.markdown(
    """
This app fits a simple **linear regression** to a daily price series and
projects it forward -- but *honestly*. The key lesson: a time series must be
split by **time**, training on the past and testing on the future, and every
forecast must carry an **uncertainty band**.

- **Price History** — see the raw series and its upward trend.
- **Train & Evaluate** — fit the model, split by time, report R² and RMSE.
- **Forecast** — project the trend forward with a ± 2σ uncertainty band.

### The build pattern this project follows
"""
)

st.code(
    "data/raw/prices.csv  ->  db/build_sqlite.py  ->  SQLite (CRUD)\n"
    "SQLite table         ->  pandas.read_sql     ->  DataFrame\n"
    "DataFrame            ->  src/features.py     ->  day-index features (X, y)\n"
    "features             ->  src/model.py        ->  fitted LinearRegression\n"
    "model                ->  src/model.py        ->  forecast + uncertainty band\n"
    "forecast             ->  src/persist.py      ->  data/derived/*.csv",
    language="text",
)

prices = load_table("prices")

st.subheader("Dataset at a glance")
cols = st.columns(3)
cols[0].metric("Trading days", len(prices))
cols[1].metric("First date", str(prices["date"].min()))
cols[2].metric("Last date", str(prices["date"].max()))

st.info(
    "Start with **Price History**, then **Train & Evaluate** to score the model "
    "on unseen future days, then **Forecast** to project it forward."
)
