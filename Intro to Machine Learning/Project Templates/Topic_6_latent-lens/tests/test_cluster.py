"""Tests for the k-means clustering in ``src/cluster.py``."""

from sklearn.metrics import adjusted_rand_score

from src.cluster import FEATURES, best_k, cluster_scores, fit_kmeans, scale


def test_scale_standardizes_features(raw_shoppers):
    """Scaled features should have ~0 mean and ~1 standard deviation."""
    X = scale(raw_shoppers)
    assert X.shape == (len(raw_shoppers), len(FEATURES))
    assert abs(X.mean()) < 1e-6
    assert abs(X.std() - 1.0) < 1e-6


def test_best_k_is_three(raw_shoppers):
    """Silhouette should peak at k=3 (the number of hidden segments)."""
    scores = cluster_scores(raw_shoppers)
    assert set(scores.columns) == {"k", "inertia", "silhouette"}
    assert best_k(scores) == 3


def test_inertia_decreases_with_k(raw_shoppers):
    """Adding clusters should always shrink the within-cluster spread."""
    scores = cluster_scores(raw_shoppers).sort_values("k")
    assert scores["inertia"].is_monotonic_decreasing


def test_kmeans_recovers_the_segments(raw_shoppers):
    """With k=3, clusters should line up with the true segments (high ARI)."""
    _, labels = fit_kmeans(raw_shoppers, 3)
    assert len(labels) == len(raw_shoppers)
    ari = adjusted_rand_score(raw_shoppers["segment"], labels)
    assert ari > 0.8
