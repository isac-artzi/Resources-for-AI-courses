"""Infrastructure tests. These pass on a fresh fork — if they don't, fix the
environment before you write any NLP code.

The review round-trip lives here rather than in the contract file on purpose:
the write path is given to you, finished, and it works before your model does.
If these tests ever go red after you start editing, you have broken the audit
trail, not the NLP."""
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
    assert body["service"] == "entityfinder-api"
    # Graders check this against the live URL in your README.
    assert body["git_sha"]
    assert body["model_version"]


def test_runs_endpoint_is_reachable_and_empty_at_first(client):
    resp = client.get("/runs")
    assert resp.status_code == 200
    assert resp.json() == {"runs": []}


def test_review_queue_is_reachable_and_empty_at_first(client):
    resp = client.get("/review_queue")
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"] == []
    assert 0.0 <= body["threshold"] <= 1.0


def test_review_queue_surfaces_a_low_confidence_prediction(client, seeded_entity):
    resp = client.get("/review_queue", params={"threshold": 0.85})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    item = body["items"][0]
    assert item["entity_id"] == seeded_entity["id"]
    assert item["confidence"] < 0.85
    assert item["text"] == "Ada Lovelace"


def test_confident_predictions_stay_out_of_the_queue(client, seeded_entity):
    # Threshold below the prediction's score: nothing to review.
    resp = client.get("/review_queue", params={"threshold": 0.20})
    assert resp.status_code == 200
    assert resp.json()["count"] == 0


def test_review_of_an_unknown_entity_is_a_404(client):
    resp = client.post(
        "/review", json={"entity_id": 999, "decision": "accept", "reviewer_id": "r1"}
    )
    assert resp.status_code == 404


def test_review_write_preserves_the_original_prediction(client, seeded_entity, fake_db):
    resp = client.post(
        "/review",
        json={
            "entity_id": seeded_entity["id"],
            "decision": "correct",
            "reviewer_id": "reviewer-7",
            "corrected_type": "PER",
            "note": "person, not an organisation",
        },
    )
    assert resp.status_code == 200
    body = resp.json()

    # The correction is recorded...
    assert body["corrected_type"] == "PER"
    # ...and the model's original answer is still there, untouched.
    assert body["original_type"] == "ORG"
    assert body["original_start_char"] == seeded_entity["start_char"]
    assert body["original_confidence"] == seeded_entity["confidence"]

    # And the prediction row itself was never modified. This is the assertion
    # that stops someone "tidying up" by updating entities in place.
    stored = fake_db["entities"][0]
    assert stored["entity_type"] == "ORG"
    assert len(fake_db["reviews"]) == 1


def test_a_reviewed_entity_leaves_the_queue(client, seeded_entity):
    client.post(
        "/review",
        json={"entity_id": seeded_entity["id"], "decision": "accept", "reviewer_id": "r2"},
    )
    resp = client.get("/review_queue", params={"threshold": 0.85})
    assert resp.json()["count"] == 0

    # ...but it is still retrievable when you ask for the reviewed rows, because
    # nothing was deleted.
    resp = client.get(
        "/review_queue", params={"threshold": 0.85, "include_reviewed": True}
    )
    assert resp.json()["count"] == 1


def test_extract_is_wired_end_to_end(client, fake_db):
    """Before your code exists this asserts the 501; afterwards it asserts the 200.

    A 501 from /extract is not a bug — it is the API saying the route, the
    logging, and the response model are all fine and the NLP layer is the part
    you still owe. Once extract_entities() works, this test starts checking the
    thing that actually matters: that every returned span really indexes into
    the text that was sent.
    """
    text = "Ada Lovelace worked in London."
    resp = client.post("/extract", json={"text": text, "model": "transformer"})
    assert resp.status_code in (200, 501)

    if resp.status_code == 501:
        assert "extract_entities" in resp.json()["detail"]
        return

    body = resp.json()
    assert body["entity_count"] == len(body["entities"])
    for entity in body["entities"]:
        assert text[entity["start_char"] : entity["end_char"]] == entity["text"]
        # Logged, and therefore reviewable.
        assert entity["entity_id"] is not None
    assert len(fake_db["extractions"]) == 1
    assert len(fake_db["entities"]) == body["entity_count"]
