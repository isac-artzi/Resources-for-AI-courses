"""Smoke tests for /healthz, /version, /schema (required by the rubric)."""
from __future__ import annotations


def test_healthz_ok(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loader"] is True
    assert body["supabase"] is True


def test_version_reports_frameworks(client):
    resp = client.get("/version")
    assert resp.status_code == 200
    body = resp.json()
    assert "torch_version" in body
    assert "sklearn_version" in body


def test_schema_lists_features(client):
    resp = client.get("/schema")
    assert resp.status_code == 200
    body = resp.json()
    assert "age" in body["numeric_features"]
    assert "occupation" in body["categorical_features"]
    assert body["target_classes"] == ["<=50K", ">50K"]
