"""
The /act request-response schema test.

Includes the dimensionality-mismatch case, which must be a 422 carrying both
numbers — not a 500, and not a bare 'invalid input'. A caller that receives a
stack trace cannot tell whether to retry, reshape, or give up.
"""

from __future__ import annotations


def test_act_returns_a_valid_action(client):
    r = client.post("/act", json={"state": [0], "policy_name": "smoke_test_policy"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["action"], int)
    assert len(body["policy_sha256"]) == 64
    assert body["latency_ms"] >= 0


def test_act_rejects_an_empty_state(client):
    assert client.post("/act", json={"state": []}).status_code == 422


def test_act_rejects_a_dimensionality_mismatch_with_a_readable_422(client):
    r = client.post(
        "/act",
        json={"state": [0.1, 0.2, 0.3, 0.4], "policy_name": "smoke_test_policy"},
    )
    assert r.status_code == 422, r.text
    detail = str(r.json()["detail"])
    assert "4" in detail and "1" in detail, f"the 422 must name both dimensions: {detail}"


def test_act_on_an_unknown_policy_is_404_and_lists_what_exists(client):
    r = client.post("/act", json={"state": [0], "policy_name": "no_such_policy"})
    assert r.status_code == 404
    assert "smoke_test_policy" in str(r.json()["detail"])


def test_act_rejects_an_out_of_range_state_index(client):
    r = client.post("/act", json={"state": [9999], "policy_name": "smoke_test_policy"})
    assert r.status_code == 422


def test_policies_endpoint_lists_size_and_checksum(client):
    body = client.get("/policies").json()
    assert body["count"] >= 1
    p = body["policies"][0]
    assert p["bytes"] > 0 and len(p["sha256"]) == 64
