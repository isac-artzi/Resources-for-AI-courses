# Cam 1 — Hello, Capture

A "hello world" for the Raspberry Pi 5 AI Camera. Capture one still image,
read back its metadata, and save it with a sortable filename.

```
picamera2  ->  metadata dict  ->  formatted report + timestamped file
```

## Problem

Before any of the AIT-224 assignments, you need to know your camera is
wired up correctly, `picamera2` is installed, and you can pull metadata
(resolution, exposure, gain) off a real capture — not just call an API
and hope.

## What you'll practice

- Opening a `Picamera2` session and running a still configuration.
- Reading back `capture_metadata()` — the numbers your camera actually
  used, which are not always what you asked for (auto-exposure will
  override your requested exposure, for example).
- Building filenames that sort correctly and never collide.

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

Saved output goes to `./captures/still_<timestamp>.jpg`.

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

`src/metadata.py` has zero hardware dependencies, so these run anywhere.
`src/hardware.py` is the only file that imports `picamera2` — it's not
unit-tested, by design, since it needs the real sensor.

## Notes

This is a plumbing check, not a syllabus assignment — Topic 1 covers
pixels/matrices/color spaces conceptually; this script just gets your
capture pipeline working first.
