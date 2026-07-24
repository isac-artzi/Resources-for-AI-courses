"""Page 1 — explore the raw shopper data.

Before running any algorithm, look at the data. This page shows the shoppers,
the five numeric features clustering will use, and the six product-category
flags the association-rule miner will use.
"""

import streamlit as st

from src.cluster import FEATURES
from src.rules import BASKET_COLUMNS
from src.ui_data import load_table

st.title("Dataset")

shoppers = load_table("shoppers")

st.markdown(
    """
Each row is one shopper. There are two kinds of columns:

- **Numeric features** (age, income, spending score, visits, basket value) —
  used by k-means and PCA.
- **Basket flags** (`bought_*`, one per product category) — used by the
  association-rule miner.

The `segment` column is the *hidden* group each shopper truly belongs to. The
unsupervised algorithms never see it; we keep it only to check their work.
"""
)

st.subheader("The shoppers")
st.dataframe(shoppers.head(50), use_container_width=True)

st.subheader("Numeric features")
st.markdown("Summary statistics for the columns k-means and PCA learn from:")
st.dataframe(shoppers[FEATURES].describe().round(1), use_container_width=True)

st.subheader("How often each category is bought")
purchase_rate = shoppers[BASKET_COLUMNS].mean().rename("purchase_rate").to_frame()
purchase_rate.index = [name.removeprefix("bought_") for name in purchase_rate.index]
st.bar_chart(purchase_rate)

st.caption(
    "These purchase rates differ by segment behind the scenes — that is exactly "
    "what lets the association-rule miner find patterns later."
)
