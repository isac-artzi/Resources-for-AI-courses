"""The ONLY module in this project that talks to real camera hardware.

Run this file directly on the Raspberry Pi 5 with the AI Camera attached:

    python -m src.hardware

It captures one still image, prints a metadata summary (via
src.metadata.format_metadata), and saves the image next to this script
using a timestamped filename (via src.metadata.build_filename).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.metadata import build_filename, format_metadata

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "captures"


def capture_still() -> Path:
    from picamera2 import Picamera2  # imported here so tests never need it

    OUTPUT_DIR.mkdir(exist_ok=True)

    picam2 = Picamera2()
    config = picam2.create_still_configuration()
    picam2.configure(config)
    picam2.start()

    metadata = picam2.capture_metadata()
    filename = build_filename("still", datetime.now())
    out_path = OUTPUT_DIR / filename
    picam2.capture_file(str(out_path))
    picam2.stop()

    print(format_metadata({**config["main"], **metadata}))
    print(f"\nSaved: {out_path}")
    return out_path


if __name__ == "__main__":
    capture_still()
