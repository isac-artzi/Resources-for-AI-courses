"""Part 6: the serving logic and the data-handling policy."""

import base64
import hashlib
import io
import json

import pytest
from PIL import Image

from seesense_lab.model import SmallResNet
from seesense_lab.predict import Predictor, sha256_hex, to_db_record, top_k


def _png_bytes(color=(200, 30, 30), size=(64, 48)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_sha256_hex():
    assert sha256_hex(b"abc") == hashlib.sha256(b"abc").hexdigest()
    assert len(sha256_hex(b"")) == 64


def test_top_k_sorted_and_truncated():
    out = top_k([0.1, 0.5, 0.05, 0.3, 0.05], ["a", "b", "c", "d", "e"], k=3)
    assert [d["label"] for d in out] == ["b", "d", "a"]
    assert out[0]["prob"] == 0.5


def test_model_is_in_eval_mode():
    p = Predictor(SmallResNet())
    assert p.model.training is False
    assert not any(m.training for m in p.model.modules())


def test_prediction_is_stable_across_calls():
    """A model left in train() mode changes BatchNorm stats on every call."""
    p = Predictor(SmallResNet())
    data = _png_bytes()
    a, b = p.predict(data), p.predict(data)
    assert a["top5"] == b["top5"]


def test_result_contract():
    p = Predictor(SmallResNet(), run_id="run42")
    data = _png_bytes()
    r = p.predict(data)
    assert set(r) == {"sha256", "label", "confidence", "top5", "gradcam_png_b64", "served_by_run_id"}
    assert r["sha256"] == hashlib.sha256(data).hexdigest()
    assert len(r["top5"]) == 5
    assert 0 < sum(d["prob"] for d in r["top5"]) <= 1.0 + 1e-6
    assert r["label"] == r["top5"][0]["label"]
    png = base64.b64decode(r["gradcam_png_b64"])
    assert Image.open(io.BytesIO(png)).size == (256, 256)


def test_rejects_non_images():
    with pytest.raises(ValueError):
        Predictor(SmallResNet()).predict(b"definitely not an image")


def test_db_record_never_contains_image_data():
    p = Predictor(SmallResNet(), run_id="run42")
    r = p.predict(_png_bytes())
    row = to_db_record(r)
    assert set(row) == {"sha256", "label", "top5", "served_by_run_id"}
    assert "gradcam_png_b64" not in json.dumps(row)
