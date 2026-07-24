"""Page 2 — group shoppers with k-means.

Two questions drive this page:
    1. How many clusters should we use? (the elbow + silhouette)
    2. What do the clusters look like? (a PCA map coloured by cluster)
"""

import streamlit as st

from src.cluster import FEATURES, best_k, fit_kmeans
from src.reduce import transform_2d
from src.ui_data import get_cluster_scores, load_table

st.title("Clusters")

shoppers = load_table("shoppers")
scores = get_cluster_scores()

st.markdown(
    """
**k-means** splits the shoppers into `k` groups so that members of a group are
as similar as possible. But *we* have to choose `k`. Two clues help:

- **Elbow** — as `k` grows, the total within-cluster spread (*inertia*) always
  falls. Look for the "elbow" where it stops falling sharply.
- **Silhouette** — how well separated the clusters are (−1 to 1). Pick the `k`
  with the highest score.
"""
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Elbow (inertia)")
    st.line_chart(scores.set_index("k")["inertia"])
with col2:
    st.subheader("Silhouette")
    st.line_chart(scores.set_index("k")["silhouette"])

suggested = best_k(scores)
st.success(f"Highest silhouette is at **k = {suggested}** — a good default choice.")

st.subheader("Explore the clusters")
k = st.slider("Number of clusters (k)", min_value=2, max_value=8, value=int(suggested))
model, labels = fit_kmeans(shoppers, k)

# Project to 2-D with PCA so we can plot every shopper as one point, coloured by
# the cluster k-means assigned it to.
coords = transform_2d(shoppers)
coords["cluster"] = labels.astype(str)
st.scatter_chart(coords, x="pc1", y="pc2", color="cluster")

st.subheader("What is each cluster like?")
st.markdown("Average feature values per cluster — this is how you *name* a segment:")
profile = shoppers[FEATURES].copy()
profile["cluster"] = labels
st.dataframe(
    profile.groupby("cluster").mean().round(1),
    use_container_width=True,
)

st.caption(
    "Because this dataset was built from three hidden segments, k = 3 recovers "
    "them almost perfectly. Real data is rarely this tidy!"
)
