# Cam 3 — Control Sweep

Sweep a camera control (analogue gain, by default) across a range,
capture one frame per value, and assemble the results into a labeled
contact sheet.

```
sweep values  ->  set_controls() per value  ->  captures  ->  grid image
```

## Problem

`AeEnable`, `AnalogueGain`, `ExposureTime`, and white-balance controls
interact in ways that are much easier to *see* than to read about. This
script gives you one image that shows the whole sweep at a glance.

## What you'll practice

- Disabling auto-exposure and driving a control manually.
- Planning a grid layout for N images (reused any time you build a
  contact sheet or comparison figure — including in later assignments).
- Burning readable labels onto frames with OpenCV before saving.

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

Saved to `./captures/control_sweep.jpg`.

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

`src/sweep.py` (value generation, grid math, labels) is pure and fully
testable. Swap `GAIN_RANGE`/`STEPS` in `src/hardware.py` to sweep
`ExposureTime` or a white-balance control instead — the grid/label logic
doesn't change.
