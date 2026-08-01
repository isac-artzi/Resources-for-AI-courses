"""Smoke test for /healthz, and the contract it is required to honour."""

from __future__ import annotations


def test_healthz_answers(client):
    r = client.get("/healthz")
    assert r.status_code in (200, 503)
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert isinstance(body["policy_artifact_loaded"], bool)
    assert isinstance(body["data_tier_reachable"], bool)


def test_healthz_reports_the_artifact_it_loaded(client):
    body = client.get("/healthz").json()
    assert body["policy_artifact_loaded"] is True, (
        "conftest writes a smoke-test artifact into policies/; if this fails, "
        "POLICY_DIR is pointing somewhere unexpected"
    )


def test_degraded_status_carries_a_reason(client):
    body = client.get("/healthz").json()
    if body["status"] == "degraded":
        assert body["detail"], "a degraded health check must say what is wrong"
