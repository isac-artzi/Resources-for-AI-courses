"""Numerical test: the model recovers the trigger-token classes AND its attention
localizes on the class-marker position.

Every synthetic sequence's label is decided by a single class marker placed right
after a randomly-positioned trigger token. A model that solves the task must
learn to *attend* to that position, so we assert both (a) held-out accuracy
clears a floor and (b) the per-timestep attention weight peaks on (or next to)
the marker. That is what makes the attention viz genuinely explanatory.
"""
from __future__ import annotations

import numpy as np

from api.training import predict_seq, train_attention
from shared.data import N_CLASSES, SEQ_LEN, generate_one


def test_attention_recovers_classes_and_localizes_unit():
    metrics, model_b64, loss = train_attention(
        n_rows=800, seq_len=SEQ_LEN, seed=42,
        lr=0.01, batch_size=32, epochs=12,
    )
    assert metrics["accuracy"] > 0.85
    assert metrics["macro_f1"] > 0.85
    assert loss[-1] < loss[0]  # training loss decreased
    assert isinstance(model_b64, str) and len(model_b64) > 0

    # Attention should peak on the marker (trigger_pos + 1) for most samples.
    hits = 0
    n = 40
    for seed in range(n):
        seq, _label, pos = generate_one(seq_len=SEQ_LEN, seed=1000 + seed)
        result = predict_seq(model_b64, [int(t) for t in seq])
        attn = np.asarray(result["attention"])
        assert abs(float(attn.sum()) - 1.0) < 1e-4  # weights are a distribution
        if int(attn.argmax()) in (pos, pos + 1):
            hits += 1
    assert hits / n > 0.8


def test_train_and_predict_sample_endpoint(client):
    ds = client.post(
        "/datasets", json={"name": "gt", "n_rows": 800, "seq_len": SEQ_LEN, "seed": 42}
    ).json()
    resp = client.post(
        "/train",
        json={"dataset_id": ds["id"], "lr": 0.01, "batch_size": 32, "epochs": 12},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["metrics"]["accuracy"] > 0.85
    run_id = body["run_id"]

    # Predict a generated sample and get attention weights back.
    pred = client.post(
        "/predict_sample", json={"run_id": run_id, "seed": 3, "true_class": 1}
    ).json()
    assert pred["label"] in range(N_CLASSES)
    assert 0.0 <= pred["confidence"] <= 1.0
    assert len(pred["probs"]) == N_CLASSES
    assert pred["true_class"] == "beta"
    assert len(pred["attention"]) == len(pred["sequence"])
    assert abs(sum(pred["attention"]) - 1.0) < 1e-4
    assert len(pred["sequence_sha256"]) == 64

    # The prediction was logged as metadata (hash + length only, no raw tokens).
    logged = client._store["sequence_metadata"]
    assert logged and logged[-1]["run_id"] == run_id
    assert logged[-1]["length"] == len(pred["sequence"])
    assert "sequence" not in logged[-1]  # no raw tokens persisted
