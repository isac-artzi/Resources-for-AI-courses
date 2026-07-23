"""Request-schema tests for the prediction + generation endpoints."""
from __future__ import annotations


def test_predict_sample_rejects_missing_run_id(client):
    # Missing 'run_id' -> 422 Unprocessable Entity from Pydantic validation.
    resp = client.post("/predict_sample", json={"seed": 0})
    assert resp.status_code == 422


def test_predict_rejects_too_short_sequence(client):
    # 'sequence' must have at least 2 characters (min_length=2).
    resp = client.post("/predict", json={"run_id": 1, "sequence": [3]})
    assert resp.status_code == 422


def test_predict_rejects_out_of_range_ids(client):
    # Character ids must be in [0, VOCAB_SIZE); 99 is out of range.
    resp = client.post("/predict", json={"run_id": 1, "sequence": [1, 99]})
    assert resp.status_code == 422


def test_generate_rejects_empty_prompt(client):
    # 'prompt' has min_length=1.
    resp = client.post("/generate", json={"run_id": 1, "prompt": ""})
    assert resp.status_code == 422
