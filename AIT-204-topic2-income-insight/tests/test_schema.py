"""Request-schema tests for the prediction endpoints (required by the rubric)."""
from __future__ import annotations


def test_predict_rejects_missing_fields(client):
    # Missing 'features' -> 422 Unprocessable Entity from Pydantic validation.
    resp = client.post("/predict", json={"run_id": 1})
    assert resp.status_code == 422


def test_predict_rejects_wrong_type(client):
    resp = client.post("/predict", json={"run_id": "not-an-int", "features": {}})
    assert resp.status_code == 422


def test_audit_rejects_bad_group(client):
    resp = client.get("/audit", params={"run_id": 1, "by": "not_a_feature"})
    assert resp.status_code == 422
