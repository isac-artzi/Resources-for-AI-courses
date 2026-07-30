"""The ONLY module in this project that talks to real camera hardware.

Run on the Pi with the AI Camera attached:

    python -m src.hardware

Sweeps AnalogueGain across a fixed range, captures one frame per value,
labels each frame, and assembles them into a single contact-sheet image
saved to ./captures/control_sweep.jpg.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.sweep import generate_sweep_values, grid_layout, label_for_value

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "captures"
GAIN_RANGE = (1.0, 8.0)
STEPS = 6


def run() -> None:
    import cv2
    from picamera2 import Picamera2  # imported here so tests never need it

    OUTPUT_DIR.mkdir(exist_ok=True)

    picam2 = Picamera2()
    config = picam2.create_still_configuration()
    picam2.configure(config)
    picam2.set_controls({"AeEnable": False})
    picam2.start()

    values = generate_sweep_values(*GAIN_RANGE, STEPS)
    frames = []
    for gain in values:
        picam2.set_controls({"AnalogueGain": gain})
        frame = picam2.capture_array()
        label = label_for_value("AnalogueGain", round(gain, 2))
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)
        frames.append(frame)
    picam2.stop()

    rows, cols = grid_layout(len(frames))
    h, w = frames[0].shape[:2]
    grid = np.zeros((rows * h, cols * w, frames[0].shape[2]), dtype=frames[0].dtype)
    for idx, frame in enumerate(frames):
        r, c = divmod(idx, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = frame

    out_path = OUTPUT_DIR / "control_sweep.jpg"
    cv2.imwrite(str(out_path), grid)
    print(f"Saved {len(frames)}-frame sweep grid ({rows}x{cols}) to {out_path}")


if __name__ == "__main__":
    run()
