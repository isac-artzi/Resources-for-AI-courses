"""Home page of the FeatureForge product.

This is the file Streamlit runs first (``streamlit run app.py``). It introduces
the product and explains the build pattern. The individual steps live in the
``pages/`` folder and appear automatically in the sidebar.

The whole app follows one data flow, the universal build pattern, wrapped in a
reproducible DVC pipeline:

    raw CSV  ->  SQLite (basic CRUD)  ->  pandas + sklearn  ->  Streamlit UI
                       \\__ prepare -> featurize -> train (DVC) __/
"""

import streamlit as st

from src.ui_data import load_table

st.set_page_config(page_title="FeatureForge", page_icon="🛠️", layout="wide")

st.title("🛠️ FeatureForge")
st.caption("An introductory feature-engineering & reproducible-pipeline template.")

st.markdown(
    """
This capstone takes a house-price model and makes it **better and
reproducible**. It shows two professional habits:

- **Feature engineering** — turn raw columns into better inputs (one-hot
  encoding, log transforms, age bins, ratios) and *measure* whether it helps.
- **Reproducible pipelines** — wire the steps together with **DVC** so anyone
  can rebuild the exact result from the raw data.

Pages:

- **Dataset** — the raw houses and why some columns are tricky.
- **Feature Engineering** — see the baseline vs engineered feature tables.
- **Pipeline Results** — the head-to-head cross-validation comparison.
- **Predict Price** — estimate one house with the engineered model.

### The pipeline this project follows
"""
)

st.code(
    "data/raw/houses.csv  ->  prepare   ->  data/derived/prepared.csv\n"
    "prepared.csv         ->  featurize ->  data/derived/features.csv\n"
    "features.csv         ->  train     ->  metrics.json + models/model.joblib\n"
    "\n"
    "Rebuild it all reproducibly with:  dvc repro",
    language="text",
)

houses = load_table("houses")

st.subheader("Dataset at a glance")
cols = st.columns(3)
cols[0].metric("Houses", len(houses))
cols[1].metric("Raw features", houses.shape[1] - 1)
cols[2].metric("Median price", f"${houses['price'].median():,.0f}")

st.info(
    "Feature engineering is only worth it if it *measurably* helps. This app "
    "always compares the engineered model against a plain baseline on the same "
    "data — so the improvement is a number you can defend, not a hunch."
)
