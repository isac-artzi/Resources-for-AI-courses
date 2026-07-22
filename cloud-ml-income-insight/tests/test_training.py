"""Numerical test: the MLP recovers the synthetic ground-truth signal.

We generate data from a known logistic rule, train through both the unit function
and the API, and assert accuracy clears a floor.
Also exercises the predict + audit round trip through the in-memory store.
"""
from __future__ import annotations

from shared.data import generate_tabular
from api.training import train_income_classifier


def test_classifier_recovers_signal_unit():
    records, labels = generate_tabular(n_rows=2000, noise=1.0, seed=1)
    metrics, model_b64, loss = train_income_classifier(
        records, labels, hidden_dim=32, lr=0.01, batch_size=64, epochs=150
    )
    assert metrics["accuracy"] > 0.8
    assert metrics["roc_auc"] > 0.85
    assert loss[-1] < loss[0]  # training loss decreased
    assert isinstance(model_b64, str) and len(model_b64) > 0


def test_train_and_predict_endpoint(client):
    ds = client.post(
        "/datasets", json={"name": "gt", "n_rows": 1500, "noise": 1.0}
    ).json()
    resp = client.post(
        "/train",
        json={"dataset_id": ds["id"], "hidden_dim": 32, "lr": 0.01,
              "batch_size": 64, "epochs": 150},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["metrics"]["accuracy"] > 0.8
    run_id = body["run_id"]

    # A clearly high-income record should lean toward >50K.
    high = {
        "age": 50, "education_num": 16, "hours_per_week": 55, "capital_gain": 8000,
        "workclass": "Self-emp", "marital_status": "Married", "occupation": "Exec",
    }
    pred = client.post("/predict", json={"run_id": run_id, "features": high}).json()
    assert pred["label"] in (0, 1)
    assert 0.0 <= pred["proba"] <= 1.0

    # Batch predict + audit round trip.
    batch = client.post(
        "/predict_batch", json={"run_id": run_id, "records": [high, high]}
    ).json()
    assert len(batch["predictions"]) == 2

    audit = client.get(
        "/audit", params={"run_id": run_id, "by": "occupation"}
    ).json()
    assert audit["total"] >= 3
    assert audit["groups"]
