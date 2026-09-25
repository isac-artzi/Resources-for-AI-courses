"""Part 7: the given FastAPI service, exercised with an in-memory model."""

import io

import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from seesense_lab import api  # noqa: E402
from seesense_lab.model import SmallResNet  # noqa: E402
from seesense_lab.predict import Predictor  # noqa: E402


@pytest.fixture()
def client(monkeypatch):
    predictor = Predictor(SmallResNet(), run_id="test-run")
    monkeypatch.setattr(api, "get_predictor", lambda: predictor)
    return TestClient(api.app)


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (40, 40), (10, 200, 10)).save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_multipart(client):
    r = client.post("/predict", files={"file": ("x.png", _png(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["served_by_run_id"] == "test-run" and len(body["top5"]) == 5


def test_predict_rejects_garbage(client):
    r = client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_predict_is_post_only(client):
    assert client.get("/predict").status_code == 405
