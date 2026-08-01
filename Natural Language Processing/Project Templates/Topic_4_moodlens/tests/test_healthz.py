"""Infrastructure tests. These pass on a fresh fork — if they don't, fix the
environment before you write any NLP code."""
from __future__ import annotations


def test_healthz_reports_ok(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["database"] in {"ok", "unreachable", "not_configured"}
    # False on a fresh fork, and False is correct: /healthz must not load the
    # model to answer. Render polls this endpoint.
    assert body["model_loaded"] is False


def test_version_exposes_build_identity(client):
    resp = client.get("/version")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "moodlens-api"
    # Graders check these against the live URL in your README.
    assert body["git_sha"]
    assert body["model_version"]
    assert body["base_model"]


def test_audit_endpoint_is_reachable_and_empty_at_first(client):
    resp = client.get("/audit")
    assert resp.status_code == 200
    assert resp.json() == {"predictions": [], "count": 0}


def test_audit_rejects_a_silly_limit(client):
    # 500 is the cap. The audit view is a table in a browser, not an export.
    assert client.get("/audit?limit=100000").status_code == 422
    assert client.get("/audit?limit=0").status_code == 422


def test_predict_is_wired_up_and_fails_loudly_until_you_implement_it(client, fake_db):
    """The template's promise: an unimplemented feature fails clearly, not weirdly.

    This test stays green on both sides of your work. Before you implement
    api/nlp.py it checks that the route turns NotImplementedError into a 501 that
    names the function you owe. After you implement it, it checks that a real
    prediction comes back in the documented shape AND that it was written to the
    predictions table — because "we log every prediction" is an audit
    requirement, not a nice-to-have, and a route that quietly stops logging is
    the kind of regression nobody notices by hand.
    """
    resp = client.post("/predict", json={"text": "worth every minute"})
    if resp.status_code == 501:
        assert "predict_sentiment" in resp.json()["detail"]
        return

    assert resp.status_code == 200
    body = resp.json()
    assert body["sentiment"]["label"] in {"negative", "positive"}
    assert 0.0 <= body["sentiment"]["probability_positive"] <= 1.0
    assert len(body["text_sha256"]) == 64
    assert len(fake_db["predictions"]) == 1
    logged = fake_db["predictions"][0]
    assert logged["model_version"]
    assert "text" not in logged, "the audit log must never hold the review text"
