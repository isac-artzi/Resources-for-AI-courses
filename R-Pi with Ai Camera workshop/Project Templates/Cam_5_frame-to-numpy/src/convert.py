"""Pure numpy/array helpers for cam-frame-to-numpy -- no camera required."""
from __future__ import annotations

import numpy as np


def describe_frame(frame: np.ndarray) -> dict:
    """Return a small dict summarizing a captured frame -- the kind of
    sanity check you should run on any array before feeding it to OpenCV.
    """
    return {
        "shape": frame.shape,
        "dtype": str(frame.dtype),
        "min": frame.min().item(),
        "max": frame.max().item(),
        "mean": round(frame.mean().item(), 2),
    }


def validate_frame_shape(frame: np.ndarray, expected_h: int, expected_w: int,
                          expected_channels: int | None = None) -> bool:
    """Check a frame matches an expected height/width (and optionally
    channel count). Raises ValueError with a helpful message on mismatch
    rather than silently returning False, since a bad capture config is
    a common early bug.
    """
    if frame.ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D array, got {frame.ndim}D")

    h, w = frame.shape[0], frame.shape[1]
    if (h, w) != (expected_h, expected_w):
        raise ValueError(f"expected {expected_h}x{expected_w}, got {h}x{w}")

    if expected_channels is not None:
        channels = frame.shape[2] if frame.ndim == 3 else 1
        if channels != expected_channels:
            raise ValueError(f"expected {expected_channels} channel(s), got {channels}")

    return True


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Swap the channel order of an (H, W, 3) array. OpenCV reads/writes
    BGR by default; most everything else (matplotlib, PIL, the model
    input most tutorials assume) expects RGB. This is the classic bug
    source in student pipelines that show "blue-tinted" images.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) array, got shape {frame.shape}")
    return frame[:, :, ::-1]
