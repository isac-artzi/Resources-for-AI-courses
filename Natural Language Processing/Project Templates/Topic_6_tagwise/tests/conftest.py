"""Shared pytest fixtures.

FINISHED CODE. The whole suite runs offline: the Supabase layer is replaced by an
in-memory fake, so you can develop and test the NLP layer with no cloud account
and no network. The one test that needs real credentials skips itself when they
are absent.

Markers (declared in pytest.ini):
    contract  — checks a function in api/nlp.py that you must implement.
                These FAIL until you write the code. That is the point: they are
                your to-do list. Run `pytest -m contract` to see what is left.
    network   — downloads a checkpoint or loads your fine-tuned model.
    cloud     — needs real Supabase credentials; skipped otherwise.

There is also a tiny hand-written treebank fixture below. Six sentences is not
enough to train anything, and that is deliberate: it is enough to check that your
corpus loader, your lookup table, and your metrics do what they claim, and it
runs in milliseconds. Test the plumbing on data you can read by eye; measure the
models on the real splits.
"""
from __future__ import annotations

import os

import pytest

# Six sentences of (word, universal tag) pairs. Note "book" and "flies": each
# appears with two different tags, with one tag more frequent than the other.
# That is what makes them useful — a most-frequent-tag lookup MUST get the
# minority reading wrong, and these fixtures let you prove it in a test.
TINY_TREEBANK = [
    [("The", "DET"), ("book", "NOUN"), ("was", "AUX"), ("heavy", "ADJ"), (".", "PUNCT")],
    [("She", "PRON"), ("read", "VERB"), ("the", "DET"), ("book", "NOUN"), (".", "PUNCT")],
    [("They", "PRON"), ("book", "VERB"), ("flights", "NOUN"), ("early", "ADV"), (".", "PUNCT")],
    [("Time", "NOUN"), ("flies", "VERB"), ("quickly", "ADV"), (".", "PUNCT")],
    [("The", "DET"), ("flies", "NOUN"), ("landed", "VERB"), (".", "PUNCT")],
    [("Fruit", "NOUN"), ("flies", "NOUN"), ("like", "ADP"), ("bananas", "NOUN"), (".", "PUNCT")],
]

# The same six sentences in CoNLL-U form, including the two rows that must be
# skipped: a range id (8-9) and an empty node (4.1). Comment lines start with #.
TINY_CONLLU = """\
# sent_id = 1
# text = The book was heavy.
1\tThe\tthe\tDET\tDT\t_\t2\tdet\t_\t_
2\tbook\tbook\tNOUN\tNN\t_\t4\tnsubj\t_\t_
3\twas\tbe\tAUX\tVBD\t_\t4\tcop\t_\t_
4\theavy\theavy\tADJ\tJJ\t_\t0\troot\t_\t_
5\t.\t.\tPUNCT\t.\t_\t4\tpunct\t_\t_

# sent_id = 2
# text = They don't book flights.
1\tThey\tthey\tPRON\tPRP\t_\t3\tnsubj\t_\t_
2-3\tdon't\t_\t_\t_\t_\t_\t_\t_\t_
2\tdo\tdo\tAUX\tVBP\t_\t4\taux\t_\t_
3\tn't\tnot\tPART\tRB\t_\t4\tadvmod\t_\t_
4\tbook\tbook\tVERB\tVB\t_\t0\troot\t_\t_
4.1\t_\t_\t_\t_\t_\t_\t_\t_\t_
5\tflights\tflight\tNOUN\tNNS\t_\t4\tobj\t_\t_
6\t.\t.\tPUNCT\t.\t_\t4\tpunct\t_\t_
"""


@pytest.fixture
def tiny_treebank():
    """Six sentences of (word, tag) pairs — enough to test logic, not to train."""
    return [list(sentence) for sentence in TINY_TREEBANK]


@pytest.fixture
def tiny_conllu_file(tmp_path):
    """A two-sentence CoNLL-U file on disk, with a range id and an empty node."""
    path = tmp_path / "tiny.conllu"
    path.write_text(TINY_CONLLU, encoding="utf-8")
    return str(path)


@pytest.fixture
def fake_db(monkeypatch):
    """Replace api.db with an in-memory store. Returns the two stores for assertions."""
    from api import db

    runs: list[dict] = []
    taggings: list[dict] = []

    def insert_run(
        model,
        tagset,
        hyperparameters,
        model_version,
        accuracy=None,
        macro_f1=None,
        metrics=None,
        notes=None,
    ):
        row = {
            "id": len(runs) + 1,
            "model": model,
            "tagset": tagset,
            "hyperparameters": hyperparameters,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "metrics": metrics or {},
            "model_version": model_version,
            "notes": notes,
            "created_at": "2026-01-01T00:00:00Z",
        }
        runs.append(row)
        return row

    def insert_tagging(
        sentence_sha256,
        token_count,
        tag_sequence,
        model,
        model_version,
        unknown_count=0,
    ):
        row = {
            "id": len(taggings) + 1,
            "sentence_sha256": sentence_sha256,
            "token_count": token_count,
            "tag_sequence": list(tag_sequence),
            "model": model,
            "model_version": model_version,
            "unknown_count": unknown_count,
            "created_at": "2026-01-01T00:00:00Z",
        }
        taggings.append(row)
        return row

    monkeypatch.setattr(db, "configured", lambda: True)
    monkeypatch.setattr(db, "ping", lambda: True)
    monkeypatch.setattr(db, "insert_run", insert_run)
    monkeypatch.setattr(db, "insert_tagging", insert_tagging)
    monkeypatch.setattr(db, "latest_runs", lambda limit=50: list(reversed(runs))[:limit])
    monkeypatch.setattr(
        db, "latest_taggings", lambda limit=100: list(reversed(taggings))[:limit]
    )
    return {"runs": runs, "taggings": taggings}


@pytest.fixture
def client(fake_db):
    """A FastAPI TestClient with the database faked out."""
    from fastapi.testclient import TestClient

    from api.main import app

    return TestClient(app)


@pytest.fixture(scope="session")
def cloud_credentials():
    """Real Supabase credentials, or a skip."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not (url and key):
        pytest.skip("SUPABASE_URL / SUPABASE_SERVICE_KEY not set — skipping cloud test.")
    return url, key
