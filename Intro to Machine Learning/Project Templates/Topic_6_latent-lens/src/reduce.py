"""Reduce many features to two with Principal Component Analysis (PCA).

Our shoppers have five numeric features, but a screen only has two axes. PCA
finds new axes -- **principal components** -- that are combinations of the
original features, ordered so the first captures the most variation, the second
the next most, and so on. Keeping the first two lets us draw every shopper as a
single point and *see* the clusters.

As with k-means we scale first, so no single large-scale feature (income) hijacks
the components.
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.cluster import FEATURES


def fit_pca(df: pd.DataFrame, n_components: int = 2) -> PCA:
    """Fit PCA on the scaled numeric features and return the fitted model."""
    X = StandardScaler().fit_transform(df[FEATURES])
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X)
    return pca


def explained_variance(pca: PCA) -> pd.DataFrame:
    """Report how much variation each component explains.

    Returns one row per component with the fraction of the total variation it
    captures and the running (cumulative) total -- a quick way to see how much
    information the first two components keep.
    """
    ratios = pca.explained_variance_ratio_
    return pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(len(ratios))],
            "explained_variance": ratios,
            "cumulative": ratios.cumsum(),
        }
    )


def transform_2d(df: pd.DataFrame) -> pd.DataFrame:
    """Project every shopper onto the first two components.

    Returns a DataFrame with columns ``pc1`` and ``pc2`` -- ready to scatter-plot.
    """
    X = StandardScaler().fit_transform(df[FEATURES])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    return pd.DataFrame({"pc1": coords[:, 0], "pc2": coords[:, 1]})
