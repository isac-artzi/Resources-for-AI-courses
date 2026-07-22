"""Request-schema tests for the prediction endpoints (required by the rubric)."""
from __future__ import annotations


def test_predict_sample_rejects_missing_run_id(client):
    # Missing 'run_id' -> 422 Unprocessable Entity from Pydantic validation.
    resp = client.post("/predict_sample", json={"seed": 0})
    assert resp.status_code == 422


def test_predict_sample_rejects_wrong_type(client):
    resp = client.post("/predict_sample", json={"run_id": "not-an-int", "seed": 0})
    assert resp.status_code == 422


def test_predict_upload_requires_a_file(client):
    # Multipart form without the 'file' part -> 422.
    resp = client.post("/predict", data={"run_id": 1})
    assert resp.status_code == 422
