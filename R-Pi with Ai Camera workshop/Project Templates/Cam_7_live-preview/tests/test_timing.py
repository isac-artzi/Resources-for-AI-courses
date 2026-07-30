import pytest

from src.timing import countdown_ticks, format_elapsed


def test_format_elapsed_under_a_minute():
    assert format_elapsed(45) == "0:45"


def test_format_elapsed_over_a_minute():
    assert format_elapsed(125) == "2:05"


def test_format_elapsed_zero():
    assert format_elapsed(0) == "0:00"


def test_format_elapsed_rejects_negative():
    with pytest.raises(ValueError):
        format_elapsed(-1)


def test_countdown_ticks_ends_at_zero():
    ticks = countdown_ticks(5)
    assert ticks == [5, 4, 3, 2, 1, 0]


def test_countdown_ticks_custom_interval():
    ticks = countdown_ticks(10, tick_interval_s=5)
    assert ticks == [10, 5, 0]


def test_countdown_ticks_zero_duration_still_includes_zero():
    assert countdown_ticks(0) == [0]


def test_countdown_ticks_rejects_negative_duration():
    with pytest.raises(ValueError):
        countdown_ticks(-1)


def test_countdown_ticks_rejects_bad_interval():
    with pytest.raises(ValueError):
        countdown_ticks(5, tick_interval_s=0)
