import pytest

from src.sweep import generate_sweep_values, grid_layout, label_for_value


def test_generate_sweep_values_endpoints():
    values = generate_sweep_values(1.0, 8.0, 6)
    assert values[0] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(8.0)
    assert len(values) == 6


def test_generate_sweep_values_rejects_too_few_steps():
    with pytest.raises(ValueError):
        generate_sweep_values(0.0, 1.0, 1)


def test_grid_layout_square_ish():
    assert grid_layout(6, max_cols=4) == (2, 4)


def test_grid_layout_respects_max_cols():
    assert grid_layout(10, max_cols=4) == (3, 4)


def test_grid_layout_single_image():
    assert grid_layout(1) == (1, 1)


def test_grid_layout_rejects_zero():
    with pytest.raises(ValueError):
        grid_layout(0)


def test_label_for_value_formats_whole_floats_as_int():
    assert label_for_value("AnalogueGain", 4.0) == "AnalogueGain=4"


def test_label_for_value_keeps_decimals():
    assert label_for_value("AnalogueGain", 2.5) == "AnalogueGain=2.5"
