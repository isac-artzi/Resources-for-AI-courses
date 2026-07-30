import pytest

from src.fps import compute_fps, format_summary, summarize_runs


def test_compute_fps_basic():
    # 30 evenly spaced frames over 1 second -> ~29 fps between first/last
    timestamps = [i * (1 / 30) for i in range(31)]
    assert compute_fps(timestamps) == pytest.approx(30.0, rel=0.05)


def test_compute_fps_needs_two_timestamps():
    with pytest.raises(ValueError):
        compute_fps([1.0])


def test_compute_fps_rejects_non_increasing():
    with pytest.raises(ValueError):
        compute_fps([1.0, 1.0])


def test_summarize_runs():
    runs = {
        "640x480": [0.0, 0.1, 0.2, 0.3],
        "1280x720": [0.0, 0.2, 0.4, 0.6],
    }
    summary = summarize_runs(runs)
    assert summary["640x480"]["frame_count"] == 4
    assert summary["640x480"]["fps"] == pytest.approx(10.0)
    assert summary["1280x720"]["fps"] == pytest.approx(5.0)


def test_format_summary_contains_all_labels():
    summary = {"640x480": {"fps": 30.0, "frame_count": 60, "elapsed_s": 2.0}}
    text = format_summary(summary)
    assert "640x480" in text
    assert "30.0" in text


def test_format_summary_empty():
    assert "no runs" in format_summary({})
