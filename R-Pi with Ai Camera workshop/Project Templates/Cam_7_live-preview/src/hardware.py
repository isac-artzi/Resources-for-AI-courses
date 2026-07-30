"""The ONLY module in this project that talks to real camera hardware.

Run on the Pi WITH A MONITOR ATTACHED (HDMI or the official touchscreen --
this will not work over a plain SSH terminal, since it opens a real
window on the Pi's own display):

    python -m src.hardware

Opens a live preview window for PREVIEW_SECONDS, printing a countdown in
the terminal, then closes the window and saves one still frame (captured
while the preview was running) to ./captures/preview_snapshot.jpg.
"""
from __future__ import annotations

import time
from pathlib import Path

from src.timing import countdown_ticks, format_elapsed

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "captures"
PREVIEW_SECONDS = 20


def run() -> None:
    from picamera2 import Picamera2, Preview  # imported here so tests never need it

    OUTPUT_DIR.mkdir(exist_ok=True)

    picam2 = Picamera2()
    config = picam2.create_preview_configuration()
    picam2.configure(config)

    # QTGL is a hardware-accelerated window on the Pi's own display.
    # If you're SSH'd in with no monitor attached, this call will fail --
    # that's expected; this script needs a physical screen.
    picam2.start_preview(Preview.QTGL)
    picam2.start()

    print(f"Live preview open for {PREVIEW_SECONDS}s -- look at the Pi's screen.")
    print("Press Ctrl+C to stop early.")

    try:
        for remaining in countdown_ticks(PREVIEW_SECONDS):
            print(f"  {format_elapsed(remaining)} remaining", end="\r")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped early.")

    out_path = OUTPUT_DIR / "preview_snapshot.jpg"
    picam2.capture_file(str(out_path))
    print(f"\nSaved a snapshot from the preview session to {out_path}")

    picam2.stop_preview()
    picam2.stop()


if __name__ == "__main__":
    run()
