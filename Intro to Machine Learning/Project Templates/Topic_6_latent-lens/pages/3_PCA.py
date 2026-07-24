"""Page 3 — reduce five features to two with PCA.

PCA finds new axes (principal components) that capture as much of the data's
variation as possible. Keeping the first two lets us draw the whole dataset on a
flat chart. Here we colour the points by the *true* segment to show that the two
components really do separate the groups.
"""

import streamlit as st

from src.reduce import explained_variance, fit_pca, transform_2d
from src.ui_data import load_table

st.title("PCA — a 2-D map of the shoppers")

shoppers = load_table("shoppers")

st.markdown(
    """
Our shoppers live in a **five-dimensional** feature space — impossible to draw
directly. **Principal Component Analysis** builds new axes, ordered by how much
variation they capture, and we keep just the first two.
"""
)

pca = fit_pca(shoppers)
variance = explained_variance(pca)

st.subheader("How much variation does each component keep?")
st.dataframe(variance.round(3), use_container_width=True)
kept = variance["cumulative"].iloc[-1]
st.success(
    f"The first two components together keep **{kept:.0%}** of all the variation "
    "— so a 2-D plot loses very little information."
)

st.subheader("Every shopper on two axes")
coords = transform_2d(shoppers)
coords["segment"] = shoppers["segment"].astype(str)
st.scatter_chart(coords, x="pc1", y="pc2", color="segment")

st.caption(
    "Each colour is a hidden segment. They land in separate regions of the map, "
    "which is exactly why k-means on the same features finds them so cleanly."
)
