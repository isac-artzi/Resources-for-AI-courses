import pytest

from src.clip import clip_duration, frame_index_at_time, sample_plan


def test_clip_duration():
    assert clip_duration(150, 30) == pytest.approx(5.0)


def test_clip_duration_rejects_bad_fps():
    with pytest.raises(ValueError):
        clip_duration(150, 0)


def test_clip_duration_rejects_negative_frames():
    with pytest.raises(ValueError):
        clip_duration(-1, 30)


def test_frame_index_at_time():
    assert frame_index_at_time(30, 2.0) == 60


def test_frame_index_at_time_rejects_negative_seconds():
    with pytest.raises(ValueError):
        frame_index_at_time(30, -1)


def test_sample_plan_includes_first_and_last():
    plan = sample_plan(total_frames=100, num_samples=5)
    assert plan[0] == 0
    assert plan[-1] == 99
    assert len(plan) == 5


def test_sample_plan_single_sample_returns_first_frame():
    assert sample_plan(total_frames=100, num_samples=1) == [0]


def test_sample_plan_caps_at_total_frames():
    plan = sample_plan(total_frames=3, num_samples=10)
    assert plan == [0, 1, 2]


def test_sample_plan_rejects_zero_total_frames():
    with pytest.raises(ValueError):
        sample_plan(total_frames=0, num_samples=3)
