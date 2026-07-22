"""Numerical test: the trained slope recovers the synthetic ground truth.

We generate data from a known line, train through the API, and assert the fitted
slope/intercept land within tolerance.
"""
from __future__ import annotations

from shared.data import generate_linear
from api.training import train_linear_regression


def test_slope_recovery_unit():
    xs, ys = generate_linear(slope=3.0, intercept=-2.0, noise=0.5, n_points=1000)
    metrics, weights, loss = train_linear_regression(
        xs, ys, lr=0.01, batch_size=64, epochs=300
    )
    assert abs(weights["slope"] - 3.0) < 0.2
    assert abs(weights["intercept"] - (-2.0)) < 0.3
    assert metrics["r2"] > 0.9
    # Loss should decrease overall.
    assert loss[-1] < loss[0]


def test_train_endpoint_recovers_slope(client):
    ds = client.post(
        "/datasets",
        json={"name": "gt", "slope": 4.0, "intercept": 0.5, "noise": 0.5, "n_points": 800},
    ).json()
    resp = client.post(
        "/train",
        json={"dataset_id": ds["id"], "lr": 0.01, "batch_size": 64, "epochs": 300},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert abs(body["weights"]["slope"] - 4.0) < 0.25
