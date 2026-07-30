# Cam 2 — Stream FPS

Measure the AI Camera's *actual* frame rate at a few resolutions, instead
of trusting the number on the spec sheet.

```
picamera2 video stream  ->  frame timestamps  ->  FPS table
```

## Problem

Requested FPS and delivered FPS are not the same thing — sensor mode,
resolution, exposure time, and USB/CSI bandwidth all affect real
throughput. Later assignments (Topic 6 motion tracking, Topic 8
compression) care about real frame timing, so it's worth measuring
directly, early.

## What you'll practice

- Opening a video-mode stream and pulling frames with `capture_array()`.
- Timing frames with `time.monotonic()` (never `time.time()` for
  intervals — it can jump backwards on clock sync).
- Turning a list of timestamps into an FPS number, and comparing across
  resolutions.

## Quick start (on the Pi)

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

Prints a comparison table across 640x480 / 1280x720 / full resolution.

## Running the tests (on any machine, no camera needed)

On the Pi, reuse the same venv:

```bash
source .venv/bin/activate   # skip if you're on a laptop with no PEP 668 lock
pip install -r requirements-dev.txt
pytest -q
```

On a laptop (no Pi, no PEP 668 restriction), a plain `pip install -r
requirements-dev.txt` works fine too -- these tests don't need
`picamera2` at all.

## Notes

`src/fps.py` is pure math over a list of floats — test it with any
synthetic timestamp list. `src/hardware.py` is the only file that talks
to the sensor.
