from datetime import datetime

import pytest

from src.metadata import build_filename, format_metadata


def test_build_filename_format():
    ts = datetime(2026, 7, 27, 9, 5, 1)
    assert build_filename("still", ts) == "still_20260727_090501.jpg"


def test_build_filename_custom_ext_with_dot():
    ts = datetime(2026, 1, 1, 0, 0, 0)
    assert build_filename("frame", ts, ext=".png") == "frame_20260101_000000.png"


def test_build_filename_rejects_empty_prefix():
    with pytest.raises(ValueError):
        build_filename("", datetime.now())


def test_format_metadata_includes_known_fields():
    config = {
        "size": (2028, 1520),
        "format": "RGB888",
        "ExposureTime": 20000,
        "AnalogueGain": 1.5,
        "Lux": 320.4,
    }
    report = format_metadata(config)
    assert "2028 x 1520" in report
    assert "RGB888" in report
    assert "20000 us" in report
    assert "1.50x" in report
    assert "320.4" in report


def test_format_metadata_handles_unknown_dict():
    report = format_metadata({"SomeUnrelatedKey": 42})
    assert "no recognized metadata fields" in report
