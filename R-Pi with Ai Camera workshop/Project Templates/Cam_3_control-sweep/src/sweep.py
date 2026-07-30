"""Pure planning helpers for cam-control-sweep -- no camera hardware required."""
from __future__ import annotations


def generate_sweep_values(start: float, stop: float, steps: int) -> list[float]:
    """Evenly spaced control values from start to stop (inclusive), steps count."""
    if steps < 2:
        raise ValueError("steps must be >= 2 to include both endpoints")
    span = stop - start
    return [start + span * i / (steps - 1) for i in range(steps)]


def grid_layout(n_images: int, max_cols: int = 4) -> tuple[int, int]:
    """Return (rows, cols) for laying out n_images in a roughly square grid,
    never exceeding max_cols columns.
    """
    if n_images < 1:
        raise ValueError("n_images must be >= 1")
    cols = min(max_cols, n_images)
    rows = -(-n_images // cols)  # ceil division
    return rows, cols


def label_for_value(control_name: str, value: float) -> str:
    """Short label to burn into each grid cell, e.g. 'ExposureTime=12500'."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{control_name}={value}"
