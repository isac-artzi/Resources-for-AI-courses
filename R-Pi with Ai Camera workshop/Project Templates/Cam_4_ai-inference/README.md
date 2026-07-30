# Cam 4 — AI Inference (on-sensor)

Run the AI Camera's on-sensor (IMX500 NPU) object-detection model and draw
the results — so you see the on-chip inference path, distinct from
running a model on the Pi's own CPU.

```
IMX500 NPU inference  ->  raw output tensors  ->  decoded detections  ->  boxes drawn on frame
```

## Problem

The headline feature of the Pi 5 AI Camera is that inference happens
*on the sensor module itself*, not on the Pi's CPU. Before building
anything on top of that, confirm the model loads, results come back in
a shape you understand, and you can map detection coordinates back onto
a full-size frame.

## What you'll practice

- Loading a model with `picamera2.devices.IMX500`, and reading its
  embedded `NetworkIntrinsics` (a model file carries its own metadata —
  task type, label list, bbox coordinate order — you don't guess it).
- Decoding raw NPU output tensors into boxes/scores/classes, including
  the fact that different model families (SSD-style vs. nanodet-style)
  decode differently, and the model tells you which one it is.
- The correct way to map a detection box back onto your frame:
  `imx500.convert_inference_coords()`, not manual arithmetic.

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

Saved to `./captures/inference.jpg`. `src/hardware.py` auto-detects
which `.rpk` model file is actually installed at
`/usr/share/imx500-models/` rather than assuming a specific filename —
if none are present, it'll tell you to run `sudo apt install imx500-all`.

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

This is a hardware-capability check, not a course assignment — none of
the syllabus topics require the on-sensor NPU path specifically, but
Topic 8's DQ on AI-driven techniques is a natural place to reference
what you saw here.

The postprocessing logic in `src/hardware.py` is adapted directly from
Raspberry Pi's own reference example (simplified to a single-shot
capture instead of a live preview loop):
https://github.com/raspberrypi/picamera2/blob/main/examples/imx500/imx500_object_detection_demo.py

