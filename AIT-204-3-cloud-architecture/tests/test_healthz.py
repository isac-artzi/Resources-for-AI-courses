"""Smoke test for /healthz (required by the rubric)."""
from __future__ import annotations


def test_healthz_ok(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loader"] is True
    assert body["supabase"] is True


def test_version_reports_torch(client):
    resp = client.get("/version")
    assert resp.status_code == 200
    assert "torch_version" in resp.json()
