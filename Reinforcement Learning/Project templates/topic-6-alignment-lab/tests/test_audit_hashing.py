"""
The audit log stores a HASH of the user's text, and never the text.

Every previous topic in this course has carried the same comment in
`db/migrations/001_init.sql` — "state is stored as a HASH, never raw: in Topic 6
the state is user text" — as a hypothetical. This is the topic where it stops
being one, and a comment is not a control. This file is the control.

WHAT COULD GO WRONG WITHOUT IT
------------------------------
`POST /score` takes arbitrary text from whoever can reach the endpoint. A
handler that logged `req.text` alongside the score would be a perfectly
reasonable-looking four-line change; the endpoint would still pass every schema
test, the Streamlit tab would still work, and the product would be retaining
user content in a database with an anon read policy on it. Nothing else in this
repository would notice.

WHAT A HASH IS AND IS NOT
-------------------------
This is pseudonymisation, not anonymisation, and the model card must say so.
Anyone holding a candidate text can confirm that it was submitted, by hashing
it themselves. That is a FEATURE for deduplication and abuse investigation and
a RISK for a short, guessable input space — "yes" hashes to the same digest for
everyone who types it. If your threat model includes an adversary who can
enumerate plausible inputs, a hash is not enough and you need a keyed MAC or no
log at all.
"""

from __future__ import annotations

import hashlib

from shared.store import get_store

SECRET = "the quick brown fox jumped over an unusually specific lazy dog 8f2a"


def test_score_writes_a_digest_and_not_the_text(client):
    r = client.post("/score", json={"text": SECRET})
    assert r.status_code == 200, r.text
    digest = r.json()["text_sha256"]
    assert digest == hashlib.sha256(SECRET.encode("utf-8")).hexdigest()

    rows = getattr(get_store(), "audit", None)
    assert rows is not None, (
        "this assertion reads MemoryStore.audit directly. Against a configured "
        "Supabase project, run the equivalent query by hand at least once:\n"
        "    select state_hash from audit_log order by created_at desc limit 5;"
    )
    mine = [row for row in rows if row.get("endpoint") == "/score"]
    assert mine, "POST /score wrote no audit row at all"

    row = mine[-1]
    assert row["state_hash"] == digest
    # The whole point, stated as an assertion over the ENTIRE row rather than
    # over the one field we expect the text to be in. A future contributor
    # adding `"text": req.text` under a different key would slip past a check
    # that only looked at `state_hash`.
    blob = " ".join(str(v) for v in row.values())
    assert SECRET not in blob, (
        f"the raw text appears in the audit row: {row!r}. Log the digest, never "
        "the content — see api/main.py::_hash_text."
    )
    for fragment in ("quick brown fox", "lazy dog", "8f2a"):
        assert fragment not in blob


def test_the_digest_is_the_full_sha256_not_a_prefix(client):
    """64 hex characters, deliberately, where /act truncates to 16.

    A truncated 64-bit digest is fine for a four-float observation vector, where
    a collision costs one mislabelled audit row. It is not fine for user text: a
    64-bit space collides by birthday at around four billion entries, and an
    audit log is exactly the artifact you do not want to have to explain a
    collision in. The extra 48 characters cost nothing.
    """
    digest = client.post("/score", json={"text": "anything at all"}).json()["text_sha256"]
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_the_same_text_produces_the_same_digest(client):
    """Deduplication must work, which is the legitimate reason to log anything here."""
    a = client.post("/score", json={"text": "repeat me"}).json()["text_sha256"]
    b = client.post("/score", json={"text": "repeat me"}).json()["text_sha256"]
    assert a == b
    c = client.post("/score", json={"text": "repeat me."}).json()["text_sha256"]
    assert a != c, "a one-character change must change the digest"


def test_compare_also_hashes_both_texts(client):
    """/compare takes two texts and must not leak either.

    Easy to forget: the obvious implementation logs `preferred` and the two
    scores, and someone later adds the texts "for debugging". The pair is
    hashed as a unit with a NUL separator, so that ("ab", "c") and ("a", "bc")
    do not collide.
    """
    left, right = "alpha bravo charlie", "delta echo foxtrot"
    client.post("/compare", json={"text_a": left, "text_b": right})
    rows = [r for r in get_store().audit if r.get("endpoint") == "/compare"]
    assert rows
    blob = " ".join(str(v) for v in rows[-1].values())
    assert "bravo" not in blob and "echo" not in blob
    assert rows[-1]["state_hash"] == hashlib.sha256(
        f"{left}\x00{right}".encode("utf-8")
    ).hexdigest()


def test_the_audit_row_still_identifies_the_artifact(client):
    """Hashing the input must not cost you attributability.

    The point of the audit log is to answer "which artifact produced this
    score" six weeks later. Dropping the text is fine; dropping the policy
    checksum would make the log decorative.
    """
    body = client.post("/score", json={"text": "attributable please"}).json()
    row = [r for r in get_store().audit if r.get("endpoint") == "/score"][-1]
    assert row["policy_sha256"] == body["policy_sha256"]
    assert len(row["policy_sha256"]) == 64
