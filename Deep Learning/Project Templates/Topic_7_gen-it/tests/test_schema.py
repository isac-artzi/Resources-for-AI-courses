"""Request-schema tests for the generative endpoints."""
from __future__ import annotations


def test_generate_rejects_missing_run_id(client):
    # Missing 'run_id' -> 422 Unprocessable Entity from Pydantic validation.
    resp = client.post("/generate", json={"z": [0.0, 0.0]})
    assert resp.status_code == 422


def test_reconstruct_rejects_wrong_type(client):
    resp = client.post("/reconstruct", json={"run_id": "not-an-int", "seed": 0})
    assert resp.status_code == 422


def test_interpolate_rejects_too_few_steps(client):
    # steps has ge=2 -> a single step is invalid.
    resp = client.post(
        "/interpolate", json={"run_id": 1, "seed_a": 0, "seed_b": 1, "steps": 1}
    )
    assert resp.status_code == 422
