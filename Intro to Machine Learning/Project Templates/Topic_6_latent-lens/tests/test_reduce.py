"""Tests for the PCA dimensionality reduction in ``src/reduce.py``."""

from src.reduce import explained_variance, fit_pca, transform_2d


def test_pca_keeps_most_variance(raw_shoppers):
    """The first two components should capture the bulk of the variation."""
    pca = fit_pca(raw_shoppers)
    variance = explained_variance(pca)
    assert list(variance["component"]) == ["PC1", "PC2"]
    # Two components keep most of the information for this dataset.
    assert variance["cumulative"].iloc[-1] > 0.8


def test_explained_variance_is_ordered(raw_shoppers):
    """Components are ordered from most to least variance explained."""
    variance = explained_variance(fit_pca(raw_shoppers))
    assert variance["explained_variance"].is_monotonic_decreasing


def test_transform_2d_shape(raw_shoppers):
    """Projecting gives one (pc1, pc2) point per shopper."""
    coords = transform_2d(raw_shoppers)
    assert list(coords.columns) == ["pc1", "pc2"]
    assert len(coords) == len(raw_shoppers)
