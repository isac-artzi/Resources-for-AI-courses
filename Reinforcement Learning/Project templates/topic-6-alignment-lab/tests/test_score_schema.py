"""
The /score, /compare and /completions contracts, driven over HTTP.

These are the endpoints the product brief adds, and the schema test is required
by the build step. Everything here goes through the test client rather than
calling the handler functions directly, because half the contract — status
codes, validation, JSON serialisation — only exists at the HTTP boundary.
"""

from __future__ import annotations

import math


def test_score_returns_the_documented_shape(client):
    r = client.post("/score", json={"text": "specific measured evidence with a tested baseline"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["reward"], float)
    assert math.isfinite(body["reward"]), "a non-finite reward would fail JSON serialisation"
    assert len(body["policy_sha256"]) == 64
    assert len(body["text_sha256"]) == 64
    assert body["tokens"] >= 1
    assert 0.0 <= body["oov_rate"] <= 1.0
    assert body["latency_ms"] >= 0


def test_score_rejects_an_empty_body_with_422(client):
    """`min_length=1` on the field, so FastAPI answers before the handler runs.

    A 422 rather than a 200 with reward 0.0. Scoring the empty string produces
    the zero vector, and the head would return `b1` — a confident number about
    nothing. Refusing is the honest contract.
    """
    assert client.post("/score", json={"text": ""}).status_code == 422


def test_score_rejects_an_oversized_body_with_422(client):
    """The 8,000-character cap is a security control, not a nicety.

    The TF-IDF transform is linear in input length and this endpoint is public;
    without a bound, one caller pasting a novel occupies a worker for as long
    as they like. The bound is declared on the Pydantic field so it is enforced
    before any of this repository's code runs.
    """
    r = client.post("/score", json={"text": "x" * 8001})
    assert r.status_code == 422


def test_score_reports_out_of_vocabulary_text_honestly(client):
    """Text the head cannot see must come back with a high OOV rate.

    Without this field a caller cannot distinguish "scored low" from "not
    really scored". The head returns its bias term for an all-OOV input, which
    is a stable, confident and completely uninformative number.
    """
    r = client.post("/score", json={"text": "qqzz yyww vvuu ttss rrqq ppoo"})
    assert r.status_code == 200, r.text
    assert r.json()["oov_rate"] > 0.8


def test_score_on_an_unknown_policy_is_404_and_lists_what_exists(client):
    r = client.post("/score", json={"text": "hello there", "policy_name": "no_such_head"})
    assert r.status_code == 404
    assert "no_such_head" in str(r.json()["detail"])


def test_score_against_a_control_policy_is_a_readable_422(client):
    """Asking a Q-table to score text is a caller error, not a server error.

    500 would be the lazy answer and tells the caller nothing. 422 naming the
    artifact's kind tells them exactly which artifact to ask instead.
    """
    r = client.post("/score", json={"text": "hello there", "policy_name": "smoke_test_policy"})
    assert r.status_code == 422
    detail = str(r.json()["detail"])
    assert "reward head" in detail and "tabular" in detail


def test_compare_returns_the_preferred_text_and_a_calibrated_probability(client):
    good = "specific measured evidence with a tested baseline and a stated caveat"
    bad = "vague hype obviously trivially whatever guaranteed flawless"
    r = client.post("/compare", json={"text_a": good, "text_b": bad})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["preferred"] in ("a", "b")
    assert body["margin"] >= 0.0, "the margin is defined as winner minus loser"
    assert 0.0 <= body["probability"] <= 1.0
    # sigmoid(margin) is the Bradley-Terry probability, and the margin is
    # non-negative by construction, so the probability can never be below 0.5.
    assert body["probability"] >= 0.5 - 1e-12
    assert body["margin"] == abs(body["reward_a"] - body["reward_b"])


def test_compare_is_symmetric_under_swapping_the_arguments(client):
    """Swapping a and b must flip the answer and leave the margin unchanged.

    This catches the whole class of bugs where the comparison accidentally
    depends on argument position rather than on the two scores — which is
    exactly the failure mode that makes a held-out accuracy of 100% look
    plausible.
    """
    a = "verified reproducible evidence quantified stepwise"
    b = "handwave magic effortless guaranteed"
    first = client.post("/compare", json={"text_a": a, "text_b": b}).json()
    second = client.post("/compare", json={"text_a": b, "text_b": a}).json()
    assert first["preferred"] != second["preferred"]
    assert abs(first["margin"] - second["margin"]) < 1e-9
    assert abs(first["reward_a"] - second["reward_b"]) < 1e-9


def test_completions_endpoint_distinguishes_empty_from_unreachable(client):
    """The UI must be able to tell "no completions" from "no database".

    Same contract as `/runs` in the base template: `degraded` is a field, not
    an empty list. A stakeholder shown an empty table cannot tell whether the
    experiment produced nothing or the database is asleep.
    """
    body = client.get("/completions").json()
    assert "degraded" in body
    assert "count" in body
    assert isinstance(body["completions"], list)


def test_completions_limit_is_capped(client):
    """An unbounded list endpoint on a free-tier instance is a denial of service."""
    r = client.get("/completions?limit=99999")
    assert r.status_code == 200
    assert len(r.json()["completions"]) <= 500


def test_alignment_runs_endpoint_answers(client):
    body = client.get("/alignment_runs").json()
    assert "degraded" in body and "runs" in body
    for run in body["runs"]:
        assert run["beta"] > 0
        if run["implicit_reward_accuracy"] is not None:
            assert 0.0 <= run["implicit_reward_accuracy"] <= 1.0


def test_version_still_reports_no_torch(client):
    """Repeated here as well as in test_no_torch.py, on purpose.

    That file asserts on `sys.modules`; this asserts on what the SERVICE tells
    a caller. Both must be true, and a product that reports one thing and does
    another has a worse problem than an extra import.
    """
    assert client.get("/version").json()["torch_imported"] is False
