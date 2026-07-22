"""Request-schema test for /predict."""
from __future__ import annotations


def test_predict_rejects_missing_fields(client):
    # Missing 'x' -> 422 Unprocessable Entity from Pydantic validation.
    resp = client.post("/predict", json={"run_id": 1})
    assert resp.status_code == 422


def test_predict_rejects_wrong_type(client):
    resp = client.post("/predict", json={"run_id": "not-an-int", "x": 3.0})
    assert resp.status_code == 422
