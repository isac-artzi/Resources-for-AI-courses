# Pi AI Camera Fundamentals

Seven small, downloadable warm-up scripts for the Raspberry Pi 5 AI Camera,
built for students entering AIT-224 (Image Processing and Computer
Vision). Each one follows the same pattern:

```
src/<pure-logic-modules>.py   -- no hardware import, unit-tested with pytest
src/hardware.py               -- the ONLY module that imports picamera2
tests/                        -- run anywhere, no camera required
README.md                     -- Problem / What you'll practice / Quick start
```

**These are not the syllabus assignments.** They're plumbing and
sanity-check scripts — students should be comfortable capturing,
streaming, sweeping controls, running on-sensor inference, converting to
numpy, and recording/reading clips *before* Topic 1's graded assignment
starts, so the syllabus assignments are about image processing concepts,
not camera-API debugging.

| Script | What it checks | Feeds into (later, syllabus) |
|---|---|---|
| [`Cam_1_hello-capture`](./Cam_1_hello-capture) | Basic still capture + metadata | Topic 1 |
| [`Cam_2_stream-fps`](./Cam_2_stream-fps) | Real-world stream throughput | Topics 6, 8 |
| [`Cam_3_control-sweep`](./Cam_3_control-sweep) | Exposure/gain behavior | Topics 2, 3 |
| [`Cam_4_ai-inference`](./Cam_4_ai-inference) | On-sensor (IMX500 NPU) inference | Topic 8 discussion (optional/bonus) |
| [`Cam_5_frame-to-numpy`](./Cam_5_frame-to-numpy) | Array shape/dtype, BGR/RGB | Topic 2 |
| [`Cam_6_clip-record-playback`](./Cam_6_clip-record-playback) | Video record + frame-by-frame read | Topic 6 |
| [`Cam_7_live-preview`](./Cam_7_live-preview) | Live display pipeline (needs a monitor on the Pi) | None directly — motivational/diagnostic |

**Priority, if time is short:** Cam 5 and Cam 6 are the closest thing to load-bearing prerequisites (Topic 2 and Topic 6 respectively assume the skills they cover). Cam 3 is the one plumbing script that also builds real conceptual intuition (visible noise from gain, previewing Topic 3). Cam 1, 2, and 7 are environment/fluency checks. Cam 4 is a showcase of the hardware's headline feature but isn't required for any graded topic — treat it as optional.

## Setup, on the Pi

Raspberry Pi OS (Bookworm+) blocks plain `pip install` at the system
level (PEP 668's "externally-managed-environment" error). The fix is a
virtual environment that can still see the apt-installed `picamera2`:

```bash
cd Cam_1_hello-capture
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.hardware
```

`picamera2` itself is **not** in any `requirements.txt` — it comes
preinstalled via apt on Raspberry Pi OS and is tied to the system's
libcamera bindings, so it should never be pip-installed fresh.
`--system-site-packages` is what lets the venv see it anyway. Each
script folder needs its own `.venv` (or reuse the same activated venv
across folders — either works).

## Running tests, on any machine (no camera needed)

```bash
cd Cam_1_hello-capture
source .venv/bin/activate   # on the Pi; skip on a laptop
pip install -r requirements-dev.txt
pytest -q
```

Every script isolates hardware access into a single `src/hardware.py`
module; everything else is pure functions over plain Python/numpy data,
so the logic can be verified without a Pi or a camera attached — useful
for students working from a laptop before their kit arrives, and for
you/the professor to sanity-check a script update without hardware.

## Topic 7 (Stereo Vision) note

None of these scripts address stereo vision yet — still pending a
decision between single-camera offset capture (default track) and a
two-camera rig (honors track). Happy to add a `Cam_8_stereo-pair-capture`
script once that's settled.

## Other gaps worth considering

- **An "environment doctor" script** — checks the venv is active,
  `picamera2` imports, the camera is detected, and lists installed
  `.rpk` files at `/usr/share/imx500-models/`, all in one shot. Given
  how much of early setup trouble tends to be environment-related (PEP
  668, missing model files), a self-diagnostic script students run
  before Day 1 would likely save the most classroom troubleshooting time
  of anything here.
- **Manual focus lock** — the AI Camera autofocuses by default, which
  can silently shift focus between exposures. Worth a short script or
  documented `AfMode`/`LensPosition` snippet before Topic 5
  (registration) and Topic 7 (stereo), both of which need a fixed, known
  geometry between shots.
