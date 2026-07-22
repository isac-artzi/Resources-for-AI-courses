"""Numerical test: the CNN recovers the synthetic shape classes + Grad-CAM works.

We generate images whose class is a known geometric
shape, train through both the unit function and the API, and assert accuracy
clears a floor. The endpoint path also exercises the predict + Grad-CAM round
trip through the in-memory store (a real pickled CNN is deserialized).
"""
from __future__ import annotations

import base64

from shared.data import IMG_SIZE, N_CLASSES
from api.training import train_cnn


def test_cnn_recovers_shape_classes_unit():
    metrics, model_b64, loss = train_cnn(
        n_rows=800, img_size=IMG_SIZE, noise=0.1, seed=42,
        lr=0.01, batch_size=32, epochs=15,
    )
    assert metrics["accuracy"] > 0.75
    assert metrics["macro_f1"] > 0.7
    assert loss[-1] < loss[0]  # training loss decreased
    assert isinstance(model_b64, str) and len(model_b64) > 0


def test_train_and_predict_sample_endpoint(client):
    ds = client.post(
        "/datasets", json={"name": "gt", "n_rows": 800, "noise": 0.1, "seed": 42}
    ).json()
    resp = client.post(
        "/train",
        json={"dataset_id": ds["id"], "lr": 0.01, "batch_size": 32, "epochs": 15},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["metrics"]["accuracy"] > 0.75
    run_id = body["run_id"]

    # Predict a generated sample and get a Grad-CAM overlay back.
    pred = client.post(
        "/predict_sample", json={"run_id": run_id, "seed": 3, "true_class": 1}
    ).json()
    assert pred["label"] in range(N_CLASSES)
    assert 0.0 <= pred["confidence"] <= 1.0
    assert len(pred["probs"]) == N_CLASSES
    assert pred["true_class"] == "vertical"
    # The PNGs are valid base64 and the raw pixels are NOT stored.
    assert base64.b64decode(pred["input_png"])
    assert base64.b64decode(pred["cam_png"])
    assert len(pred["image_sha256"]) == 64

    # The prediction was logged as metadata (hash + shape only).
    logged = client._store["image_metadata"]
    assert logged and logged[-1]["run_id"] == run_id
    assert logged[-1]["width"] == IMG_SIZE and logged[-1]["height"] == IMG_SIZE
    assert "input_png" not in logged[-1]  # no pixels persisted
