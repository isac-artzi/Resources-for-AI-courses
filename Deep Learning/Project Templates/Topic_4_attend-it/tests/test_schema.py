"""Request-schema tests for the prediction endpoints."""
from __future__ import annotations


def test_predict_sample_rejects_missing_run_id(client):
    # Missing 'run_id' -> 422 Unprocessable Entity from Pydantic validation.
    resp = client.post("/predict_sample", json={"seed": 0})
    assert resp.status_code == 422


def test_predict_sample_rejects_wrong_type(client):
    resp = client.post("/predict_sample", json={"run_id": "not-an-int", "seed": 0})
    assert resp.status_code == 422


def test_predict_rejects_too_short_sequence(client):
    # 'sequence' must have at least 2 tokens (min_length=2).
    resp = client.post("/predict", json={"run_id": 1, "sequence": [3]})
    assert resp.status_code == 422
