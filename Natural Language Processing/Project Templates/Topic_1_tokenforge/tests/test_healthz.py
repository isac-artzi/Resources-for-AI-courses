"""Infrastructure tests. These pass on a fresh fork — if they don't, fix the
environment before you write any NLP code."""
from __future__ import annotations


def test_healthz_reports_ok(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["database"] in {"ok", "unreachable", "not_configured"}


def test_version_exposes_build_identity(client):
    resp = client.get("/version")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "tokenforge-api"
    # Graders check this against the live URL in your README.
    assert body["git_sha"]
    assert body["model_version"]


def test_runs_endpoint_is_reachable_and_empty_at_first(client):
    resp = client.get("/runs")
    assert resp.status_code == 200
    assert resp.json() == {"runs": []}
