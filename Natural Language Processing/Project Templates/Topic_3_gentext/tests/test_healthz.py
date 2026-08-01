"""Infrastructure tests. These pass on a fresh fork — if they don't, fix the
environment before you write any generation code."""
from __future__ import annotations


def test_healthz_reports_ok(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["database"] in {"ok", "unreachable", "not_configured"}
    # False on a fresh process: /healthz must not load the model. A health check
    # that pulls half a gigabyte of weights times out and Render restarts you.
    assert body["model_loaded"] is False


def test_version_exposes_build_identity(client):
    resp = client.get("/version")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "gentext-api"
    # Graders check these against the live URL in your README.
    assert body["git_sha"]
    assert body["model_version"]
    assert body["base_model"]


def test_history_endpoint_is_reachable_and_empty_at_first(client):
    resp = client.get("/history")
    assert resp.status_code == 200
    assert resp.json() == {"generations": []}


def test_generate_returns_501_until_you_implement_it(client):
    """The 501 is the template being honest about what is missing.

    Once nlp.generate() works this test is no longer meaningful — delete it then,
    and let the contract tests carry the weight.
    """
    resp = client.post("/generate", json={"prompt": "hello"})
    assert resp.status_code in {200, 501}


def test_rate_rejects_an_out_of_range_score_before_it_reaches_your_code(client):
    """Pydantic validates at the edge, so nlp.record_rating never sees a 9."""
    resp = client.post(
        "/rate", json={"generation_id": 1, "rater_id": "rater-a", "rating": 9}
    )
    assert resp.status_code == 422
