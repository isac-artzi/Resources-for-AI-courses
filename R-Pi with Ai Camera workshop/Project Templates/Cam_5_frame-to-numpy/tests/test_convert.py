import numpy as np
import pytest

from src.convert import bgr_to_rgb, describe_frame, validate_frame_shape


def test_describe_frame_reports_expected_fields():
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    frame[..., 0] = 255
    info = describe_frame(frame)
    assert info["shape"] == (10, 20, 3)
    assert info["dtype"] == "uint8"
    assert info["max"] == 255
    assert info["min"] == 0


def test_validate_frame_shape_passes_on_match():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert validate_frame_shape(frame, 480, 640, expected_channels=3) is True


def test_validate_frame_shape_rejects_wrong_size():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        validate_frame_shape(frame, 720, 1280)


def test_validate_frame_shape_rejects_wrong_channels():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        validate_frame_shape(frame, 480, 640, expected_channels=1)


def test_validate_frame_shape_rejects_bad_ndim():
    frame = np.zeros((3, 3, 3, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        validate_frame_shape(frame, 3, 3)


def test_bgr_to_rgb_swaps_first_and_last_channel():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[..., 0] = 10  # B
    frame[..., 2] = 200  # R
    swapped = bgr_to_rgb(frame)
    assert swapped[0, 0, 0] == 200  # now R first
    assert swapped[0, 0, 2] == 10   # B last


def test_bgr_to_rgb_rejects_non_3_channel():
    frame = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError):
        bgr_to_rgb(frame)
