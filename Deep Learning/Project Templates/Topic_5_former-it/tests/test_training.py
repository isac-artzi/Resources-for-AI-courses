"""Numerical test: the Transformer learns the palindrome rule AND returns valid
per-head self-attention matrices.

The label of each synthetic sequence is whether it is a palindrome -- a rule that
requires comparing position ``i`` with position ``n-1-i``. A self-attention head
can express exactly that comparison, so we assert (a) held-out accuracy clears a
floor well above the 50% binary baseline and (b) the returned attention is a
stack of proper distributions (each row sums to 1) of the right shape, which is
what the UI renders as per-head heatmaps.
"""
from __future__ import annotations

import numpy as np

from api.training import predict_seq, train_transformer
from shared.data import N_CLASSES, SEQ_LEN, generate_one


def test_transformer_learns_palindrome_unit():
    metrics, model_b64, loss = train_transformer(
        n_rows=2000, seq_len=SEQ_LEN, seed=42,
        lr=0.005, batch_size=64, epochs=20,
    )
    assert metrics["accuracy"] > 0.85
    assert metrics["macro_f1"] > 0.85
    assert loss[-1] < loss[0]  # training loss decreased
    assert isinstance(model_b64, str) and len(model_b64) > 0

    # Attention comes back as [n_layers][n_heads][T][T], each row a distribution.
    seq, _label = generate_one(seq_len=SEQ_LEN, seed=7)
    result = predict_seq(model_b64, [int(t) for t in seq])
    attn = np.asarray(result["attention"])
    assert attn.shape == (
        result["n_layers"], result["n_heads"], SEQ_LEN, SEQ_LEN
    )
    row_sums = attn.sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-4)


def test_train_and_predict_sample_endpoint(client):
    ds = client.post(
        "/datasets", json={"name": "gt", "n_rows": 2000, "seq_len": SEQ_LEN, "seed": 42}
    ).json()
    resp = client.post(
        "/train",
        json={"dataset_id": ds["id"], "lr": 0.005, "batch_size": 64, "epochs": 20},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["metrics"]["accuracy"] > 0.85
    run_id = body["run_id"]

    # Predict a generated palindrome and get attention matrices back.
    pred = client.post(
        "/predict_sample", json={"run_id": run_id, "seed": 3, "true_class": 1}
    ).json()
    assert pred["label"] in range(N_CLASSES)
    assert 0.0 <= pred["confidence"] <= 1.0
    assert len(pred["probs"]) == N_CLASSES
    assert pred["true_class"] == "palindrome"
    assert len(pred["attention"]) == pred["n_layers"]
    assert len(pred["attention"][0]) == pred["n_heads"]
    assert len(pred["sequence_sha256"]) == 64

    # The prediction was logged as metadata (hash + length only, no raw symbols).
    logged = client._store["sequence_metadata"]
    assert logged and logged[-1]["run_id"] == run_id
    assert logged[-1]["length"] == len(pred["sequence"])
    assert "sequence" not in logged[-1]  # no raw symbols persisted
