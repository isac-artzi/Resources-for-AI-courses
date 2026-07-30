"""The ONLY module in this project that talks to real camera hardware.

Run on the Pi with the AI Camera attached:

    python -m src.hardware

Captures a frame straight into a numpy array (no file I/O), runs it
through the validation/description/conversion helpers in src.convert,
and prints the results.
"""
from __future__ import annotations

from src.convert import bgr_to_rgb, describe_frame, validate_frame_shape

EXPECTED_SIZE = (1520, 2028)  # (height, width)


def run() -> None:
    from picamera2 import Picamera2  # imported here so tests never need it

    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": (2028, 1520)})
    picam2.configure(config)
    picam2.start()

    frame = picam2.capture_array()
    picam2.stop()

    print("Raw capture ->", describe_frame(frame))
    validate_frame_shape(frame, *EXPECTED_SIZE, expected_channels=3)
    print("Shape check passed.")

    rgb = bgr_to_rgb(frame)
    print("After BGR->RGB swap ->", describe_frame(rgb))


if __name__ == "__main__":
    run()
