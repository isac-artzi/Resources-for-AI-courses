"""Group similar shoppers with k-means clustering.

Clustering is *unsupervised*: we never show the algorithm the true segment. We
give it only the numeric features and ask it to find groups of similar rows.

Two beginner ideas drive this module:

* **Scale first.** k-means measures straight-line (Euclidean) distance, so a
  feature on a big scale (income, in the tens of thousands) would completely
  dominate one on a small scale (visits per month). ``StandardScaler`` puts
  every feature on the same footing (mean 0, standard deviation 1) so each one
  counts equally.
* **How many clusters?** k-means needs you to pick ``k`` up front. We try a
  range of ``k`` values and look at two clues: the **elbow** (where adding more
  clusters stops shrinking the within-cluster spread much) and the
  **silhouette** score (how well separated the clusters are, from -1 to 1;
  higher is better).
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# The numeric features k-means learns from. The basket (bought_*) columns and
# the ground-truth ``segment`` are deliberately left out.
FEATURES = [
    "age",
    "annual_income",
    "spending_score",
    "visits_per_month",
    "avg_basket_value",
]


def scale(df: pd.DataFrame) -> np.ndarray:
    """Standardise the numeric features (mean 0, std 1) and return an array."""
    return StandardScaler().fit_transform(df[FEATURES])


def fit_kmeans(df: pd.DataFrame, n_clusters: int, random_state: int = 42):
    """Fit k-means on the scaled features and return ``(model, labels)``.

    ``n_init=10`` restarts the algorithm 10 times from different seeds and keeps
    the best result, which avoids unlucky starting points.
    """
    X = scale(df)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return model, labels


def cluster_scores(df: pd.DataFrame, k_values=range(2, 9)) -> pd.DataFrame:
    """Score a range of ``k`` values to help choose the number of clusters.

    Returns one row per ``k`` with:
        * ``inertia``     -- total within-cluster spread (lower is tighter);
                             used for the elbow plot.
        * ``silhouette``  -- how separated the clusters are (higher is better).
    """
    X = scale(df)
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(X, labels)),
            }
        )
    return pd.DataFrame(rows)


def best_k(scores: pd.DataFrame) -> int:
    """Return the ``k`` with the highest silhouette score."""
    return int(scores.loc[scores["silhouette"].idxmax(), "k"])
