# Cam 5 — Frame to NumPy

Capture straight into a numpy array (no file I/O), validate its
shape/dtype, and fix the BGR/RGB channel-order mismatch before it bites
you in a later assignment.

```
picamera2 capture_array()  ->  shape/dtype check  ->  BGR->RGB swap
```

## Problem

Every syllabus topic from 2 onward assumes you can get a frame into a
numpy array in the shape OpenCV/your filters expect. The single most
common early bug is a "blue-tinted" image — OpenCV defaults to BGR
channel order, but everything else (matplotlib, most tutorials, most
model inputs) expects RGB.

## What you'll practice

- Capturing directly to a numpy array instead of a file.
- Writing a shape/dtype validator that fails loudly with a useful
  message, instead of a confusing shape-mismatch error three files away.
- The BGR<->RGB swap, and why `frame[:, :, ::-1]` is how you do it.

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

All of `src/convert.py` runs on synthetic numpy arrays — no camera
needed to test it.

## Notes

This is the script to run *right before* starting the Topic 2 (Filtering
& Enhancement) assignment — it establishes exactly the array shape/dtype
your filtering code will assume.
