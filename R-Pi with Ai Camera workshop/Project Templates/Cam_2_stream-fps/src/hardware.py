"""The ONLY module in this project that talks to real camera hardware.

Run on the Pi with the AI Camera attached:

    python -m src.hardware

Streams video at a few resolutions, times a fixed number of frames at
each, and prints a comparison table via src.fps.
"""
from __future__ import annotations

import time

from src.fps import format_summary, summarize_runs

RESOLUTIONS = {
    "640x480": (640, 480),
    "1280x720": (1280, 720),
    "2028x1520": (2028, 1520),
}
FRAMES_PER_RUN = 60


def measure_resolution(picam2, size: tuple[int, int]) -> list[float]:
    config = picam2.create_video_configuration(main={"size": size})
    picam2.configure(config)
    picam2.start()

    timestamps = []
    for _ in range(FRAMES_PER_RUN):
        picam2.capture_array()
        timestamps.append(time.monotonic())

    picam2.stop()
    return timestamps


def run() -> None:
    from picamera2 import Picamera2  # imported here so tests never need it

    picam2 = Picamera2()
    runs = {}
    for label, size in RESOLUTIONS.items():
        print(f"Measuring {label} ...")
        runs[label] = measure_resolution(picam2, size)

    print()
    print(format_summary(summarize_runs(runs)))


if __name__ == "__main__":
    run()
