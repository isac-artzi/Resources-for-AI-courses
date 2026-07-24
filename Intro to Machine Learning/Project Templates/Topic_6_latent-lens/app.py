"""Home page of the LatentLens product.

This is the file Streamlit runs first (``streamlit run app.py``). It introduces
the product and explains the build pattern. The individual steps live in the
``pages/`` folder and appear automatically in the sidebar.

The whole app follows one data flow, the universal build pattern:

    raw CSV  ->  SQLite (basic CRUD)  ->  pandas + sklearn  ->  Streamlit UI
"""

import streamlit as st

from src.ui_data import load_table

st.set_page_config(page_title="LatentLens", page_icon="🔭", layout="wide")

st.title("🔭 LatentLens")
st.caption("An introductory unsupervised-learning product template.")

st.markdown(
    """
This app finds **hidden structure** in shopper data — *without* using any
labels. It does three classic unsupervised jobs:

- **Clusters** — group similar shoppers with k-means, and use the elbow and
  silhouette to decide how many groups there are.
- **PCA** — squeeze five numeric features down to two so we can plot every
  shopper as a single point and *see* the groups.
- **Association Rules** — mine "customers who buy X also buy Y" patterns from
  their shopping baskets.

The **Dataset** page shows the raw shoppers first.

### The build pattern this project follows
"""
)

st.code(
    "data/raw/shoppers.csv  ->  db/build_sqlite.py  ->  SQLite (CRUD)\n"
    "SQLite table           ->  pandas.read_sql     ->  DataFrame\n"
    "DataFrame              ->  StandardScaler       ->  scaled features\n"
    "scaled features        ->  k-means / PCA        ->  clusters + 2-D map\n"
    "baskets                ->  Apriori              ->  association rules\n"
    "results                ->  Streamlit            ->  the pages you see here",
    language="text",
)

shoppers = load_table("shoppers")

st.subheader("Dataset at a glance")
cols = st.columns(3)
cols[0].metric("Shoppers", len(shoppers))
cols[1].metric("Numeric features", 5)
cols[2].metric("Product categories", 6)

st.info(
    "Unsupervised learning has no answer key. Instead of measuring accuracy, we "
    "look for structure the algorithm can rediscover on its own — clusters that "
    "are well separated, components that capture most of the variation, and "
    "buying patterns that happen more often than chance."
)
