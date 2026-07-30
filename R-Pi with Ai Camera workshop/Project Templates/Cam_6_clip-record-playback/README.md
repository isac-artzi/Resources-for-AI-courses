# Cam 6 — Clip Record & Playback

Record a short clip with `picamera2`, then read it back frame-by-frame
with `cv2.VideoCapture` — the video I/O pattern reused in Topic 6's
motion-tracking assignment.

```
picamera2 H264Encoder  ->  .mp4 file  ->  cv2.VideoCapture  ->  sampled frames
```

## Problem

Topic 6 (motion tracking with optical flow) needs you to pull frames out
of a video source one at a time and process each one. Recording *and*
reading back a clip are two separate APIs (`picamera2` for recording,
OpenCV for reading) — get comfortable with the handoff between them
before you're also debugging Lucas-Kanade.

## What you'll practice

- Recording with `H264Encoder` + `FfmpegOutput`.
- Seeking to specific frame indices with `cv2.VideoCapture` /
  `CAP_PROP_POS_FRAMES`.
- Planning an even sample of frames across a clip (e.g., for a contact
  sheet or a quick visual QA pass).

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

Records a 5-second clip to `./captures/clip.mp4`, then saves a sampled
contact sheet to `./captures/clip_samples.jpg`.

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

`src/clip.py` (duration/index/sampling math) is pure and fully testable.
The frame-count/FPS relationship here is exactly what you'll need when
computing motion vectors between specific frames in Topic 6.
