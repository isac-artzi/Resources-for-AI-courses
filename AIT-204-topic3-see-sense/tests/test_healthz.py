"""Smoke tests for /healthz, /version, /classes (required by the rubric)."""
from __future__ import annotations

from shared.data import CLASS_NAMES, N_CLASSES


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
    body = resp.json()
    assert "torch_version" in body
    assert "git_sha" in body


def test_classes_lists_labels(client):
    resp = client.get("/classes")
    assert resp.status_code == 200
    body = resp.json()
    assert body["classes"] == CLASS_NAMES
    assert body["n_classes"] == N_CLASSES
