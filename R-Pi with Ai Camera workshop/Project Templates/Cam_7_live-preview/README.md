# Cam 7 — Live Preview Window

Open a real-time video window on the Pi's own screen, using picamera2's
built-in QTGL preview — the same mechanism behind the AI Camera's
out-of-the-box demos.

```
picamera2 start_preview(QTGL)  ->  live window on the Pi's display  ->  snapshot on exit
```

## Problem

Everything else in this series either saves a file or prints to the
terminal — nothing shows you a live feed. This script closes that gap:
a real, moving preview window, running directly on the Pi's screen (no
network, no browser, no second device needed).

## What you'll practice

- Opening a hardware-accelerated preview window with
  `picam2.start_preview(Preview.QTGL)`.
- Running camera capture and a live display at the same time.
- Capturing a still *while* the preview is running — the preview
  window and file capture aren't mutually exclusive.

## Requirement: a monitor attached to the Pi

This opens a window on the Pi's **own physical display** (HDMI monitor
or the official touchscreen). It will not work over a plain SSH
terminal with no screen attached — if you're headless, you won't see
anything (the script will still run, but there's no display to show
the window on).

## Quick start (on the Pi, with a monitor attached)

Raspberry Pi OS (Bookworm+) locks down the system Python (PEP 668), so a
plain `pip install` will refuse to run with an "externally-managed-environment"
error. `apt` doesn't read requirements.txt files, so `apt install -r ...`
isn't the fix either -- what you want is a virtual environment that can
still see the apt-installed `picamera2`:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.hardware
```

`--system-site-packages` is the key flag -- it lets this venv see the
system's `picamera2` (installed via apt, not pip) while still letting you
`pip install` the rest freely inside the venv, with no PEP 668 error.
Next time you open a terminal, just re-run `source .venv/bin/activate`
before running the script again.

The preview stays open for 20 seconds (edit `PREVIEW_SECONDS` in
`src/hardware.py` to change that), counting down in the terminal. Press
Ctrl+C to close it early. A snapshot taken during the session is saved
to `./captures/preview_snapshot.jpg`.

## Running the tests (on any machine, no camera or monitor needed)

```bash
source .venv/bin/activate   # on the Pi; skip on a laptop
pip install -r requirements-dev.txt
pytest -q
```

`src/timing.py` (the countdown display math) is pure and fully testable.
`src/hardware.py` is the only file that opens a camera or a window.

## Notes

This is a hardware/display capability check, not a course assignment.
It's a good one to run first when you're setting up a new Pi, since it
confirms the camera, `picamera2`, and the display pipeline are all
working together before you rely on any of that for graded work.
