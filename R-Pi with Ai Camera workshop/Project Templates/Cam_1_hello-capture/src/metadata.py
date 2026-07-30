"""Pure helper functions for cam-hello -- no camera hardware required.

Kept separate from src/hardware.py so these can be unit-tested on a
laptop with plain pytest, without picamera2 or a physical camera attached.
"""
from __future__ import annotations

from datetime import datetime


def build_filename(prefix: str, timestamp: datetime, ext: str = "jpg") -> str:
    """Build a sortable, collision-resistant filename.

    Example: build_filename("still", datetime(2026, 7, 27, 9, 5, 1)) ->
    'still_20260727_090501.jpg'
    """
    if not prefix:
        raise ValueError("prefix must be a non-empty string")
    if ext.startswith("."):
        ext = ext[1:]
    stamp = timestamp.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}.{ext}"


def format_metadata(config: dict) -> str:
    """Turn a picamera2-style metadata/config dict into a readable report.

    `config` is expected to look roughly like what
    Picamera2().capture_metadata() or .camera_configuration() returns --
    but this function only reads keys it recognizes, so it's safe to call
    with a hand-built dict in tests.
    """
    lines = ["Capture summary", "----------------"]

    size = config.get("size")
    if size:
        w, h = size
        lines.append(f"Resolution : {w} x {h}")

    fmt = config.get("format")
    if fmt:
        lines.append(f"Format     : {fmt}")

    exposure = config.get("ExposureTime")
    if exposure is not None:
        lines.append(f"Exposure   : {exposure} us ({exposure / 1000:.2f} ms)")

    gain = config.get("AnalogueGain")
    if gain is not None:
        lines.append(f"Gain       : {gain:.2f}x")

    lux = config.get("Lux")
    if lux is not None:
        lines.append(f"Est. lux   : {lux:.1f}")

    if len(lines) == 2:
        lines.append("(no recognized metadata fields present)")

    return "\n".join(lines)
