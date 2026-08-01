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
    assert body["service"] == "tagwise-api"
    # Graders check this against the live URL in your README.
    assert body["git_sha"]
    assert body["model_version"]


def test_runs_endpoint_is_reachable_and_empty_at_first(client):
    resp = client.get("/runs")
    assert resp.status_code == 200
    assert resp.json() == {"runs": []}


def test_tag_endpoint_exists_and_answers_before_you_implement_anything(client):
    """501 now, 200 later — either way the route is wired up correctly.

    A NotImplementedError from api/nlp.py becomes a 501 with the docstring's
    message in it, which is what the UI renders as "not implemented yet". Once
    tag_sentence works, the same request returns a tagged sentence. A 500 or a
    404 here means something is wrong with the plumbing, not with your NLP.
    """
    resp = client.post("/tag", json={"sentence": "They book the flight.", "model": "baseline"})
    assert resp.status_code in {200, 501}
    if resp.status_code == 501:
        assert resp.json()["detail"]


def test_tag_endpoint_rejects_an_unknown_model_name(client):
    """422 from pydantic, before your code runs. Validation is not your job."""
    resp = client.post("/tag", json={"sentence": "hello", "model": "magic"})
    assert resp.status_code == 422
