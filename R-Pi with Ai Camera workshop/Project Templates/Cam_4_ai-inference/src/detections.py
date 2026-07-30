"""Pure helpers for cam-ai-inference -- no camera/NPU hardware required.

The actual IMX500 postprocessing pipeline (parsing raw NPU tensors,
picking the nanodet-vs-default decode path, converting model-space
coordinates to frame-space coordinates) depends directly on picamera2's
IMX500 devlib objects and can't be made pure -- that logic lives in
src/hardware.py, adapted from Raspberry Pi's official reference example:
https://github.com/raspberrypi/picamera2/blob/main/examples/imx500/imx500_object_detection_demo.py

What CAN be pure and unit-tested here: picking which model file to load,
filtering by confidence, and formatting a label for drawing.
"""
from __future__ import annotations


def resolve_model_path(preferred: str, search_dir: str, available: list[str]) -> str:
    """Pick which .rpk model file to load.

    `available` is the list of filenames actually present in `search_dir`
    (pass this in rather than letting this function touch the filesystem,
    so it stays pure and testable). Prefers an exact match on `preferred`;
    falls back to the first available .rpk file; raises a clear error
    listing what IS there if nothing usable is found.
    """
    if preferred in available:
        return f"{search_dir}/{preferred}"

    rpk_files = sorted(f for f in available if f.endswith(".rpk"))
    if rpk_files:
        return f"{search_dir}/{rpk_files[0]}"

    raise FileNotFoundError(
        f"No .rpk model files found in {search_dir}. "
        f"Files present: {available or '(directory empty or missing)'}. "
        "Try: sudo apt install imx500-all"
    )


def filter_by_confidence(detections: list[dict], threshold: float = 0.5) -> list[dict]:
    """Keep only detections at or above `threshold`.

    Each detection is expected to have a "confidence" key -- this doesn't
    care what else is in the dict, so it works whether you built it by
    hand (in a test) or from real IMX500 output (in src/hardware.py).
    """
    return [d for d in detections if d["confidence"] >= threshold]


def label_for_category(labels: list[str], category_index: int) -> str:
    """Look up a human-readable label for a raw category index.

    Falls back to the numeric index (as a string) if `labels` is empty
    or the index is out of range -- e.g. when a model has no embedded
    label list and we haven't supplied one ourselves.
    """
    if labels and 0 <= category_index < len(labels):
        return labels[category_index]
    return str(category_index)


def format_detection_label(label: str, confidence: float) -> str:
    """Text to draw next to a bounding box, e.g. 'person (0.87)'."""
    return f"{label} ({confidence:.2f})"
