import pytest

from src.detections import (
    filter_by_confidence,
    format_detection_label,
    label_for_category,
    resolve_model_path,
)


def test_filter_by_confidence():
    detections = [
        {"label": "0", "confidence": 0.3, "box": (0, 0, 1, 1)},
        {"label": "1", "confidence": 0.8, "box": (0, 0, 1, 1)},
    ]
    kept = filter_by_confidence(detections, threshold=0.5)
    assert len(kept) == 1
    assert kept[0]["label"] == "1"


def test_resolve_model_path_prefers_exact_match():
    available = ["imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk", "imx500_network_other.rpk"]
    result = resolve_model_path(
        "imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk", "/models", available
    )
    assert result == "/models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"


def test_resolve_model_path_falls_back_to_first_rpk():
    available = ["readme.txt", "imx500_network_other.rpk"]
    result = resolve_model_path("missing.rpk", "/models", available)
    assert result == "/models/imx500_network_other.rpk"


def test_resolve_model_path_raises_with_helpful_message_when_none_found():
    with pytest.raises(FileNotFoundError, match="imx500-all"):
        resolve_model_path("missing.rpk", "/models", [])


def test_label_for_category_uses_labels_list_when_available():
    assert label_for_category(["cat", "dog", "bird"], 1) == "dog"


def test_label_for_category_falls_back_to_index_when_no_labels():
    assert label_for_category([], 7) == "7"


def test_label_for_category_falls_back_when_index_out_of_range():
    assert label_for_category(["cat", "dog"], 5) == "5"


def test_format_detection_label():
    assert format_detection_label("person", 0.873) == "person (0.87)"
