"""Infrastructure tests. These pass on a fresh fork — if they don't, fix the
environment before you write any model code."""
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
    assert body["service"] == "classify-it-api"
    # Graders check this against the live URL in your README.
    assert body["git_sha"]
    assert body["model_version"]


def test_runs_endpoint_is_reachable_and_empty_at_first(client):
    resp = client.get("/runs")
    assert resp.status_code == 200
    assert resp.json() == {"runs": []}


def test_unimplemented_nlp_surfaces_as_501_not_500(client):
    """A missing implementation is a 501, which the UI renders as a to-do.

    Once you implement nlp.predict() this becomes a 200 (or a 400 if the
    artifact is missing), which is why the assertion accepts all three. What it
    never accepts is a 500: an unhandled exception means the route lost its error
    handling, and the UI will show a stack trace to whoever is watching your demo.
    """
    resp = client.post("/predict", json={"text": "hello", "model_kind": "baseline"})
    assert resp.status_code in {200, 400, 501}
    assert resp.status_code != 500


def test_predict_rejects_empty_text_before_it_reaches_your_code(client):
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422


def test_predict_rejects_an_unknown_model_kind(client):
    # The schema allows exactly two model kinds. A typo should fail loudly at the
    # boundary rather than reaching load_model() and raising something vaguer.
    resp = client.post("/predict", json={"text": "hello", "model_kind": "bert"})
    assert resp.status_code == 422


def test_batch_endpoint_enforces_its_size_cap(client):
    resp = client.post(
        "/predict_batch", json={"texts": ["x"] * 65, "model_kind": "baseline"}
    )
    assert resp.status_code == 422


def test_schema_endpoint_exists(client):
    resp = client.get("/schema")
    assert resp.status_code in {200, 400, 501}
    assert resp.status_code != 500
