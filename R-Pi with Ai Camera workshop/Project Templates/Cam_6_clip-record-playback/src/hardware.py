"""The ONLY module in this project that talks to real camera/video hardware.

Run on the Pi with the AI Camera attached:

    python -m src.hardware

Records a short H.264 clip with picamera2, then reads it back with
cv2.VideoCapture frame-by-frame, sampling a handful of frames (via
src.clip.sample_plan) into a contact sheet saved to
./captures/clip_samples.jpg.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src.clip import clip_duration, sample_plan

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "captures"
RECORD_SECONDS = 5
FPS = 30
NUM_SAMPLES = 6


def record_clip(picam2, out_path: Path) -> None:
    from picamera2.encoders import H264Encoder
    from picamera2.outputs import FfmpegOutput

    encoder = H264Encoder()
    output = FfmpegOutput(str(out_path))
    picam2.start_recording(encoder, output)
    time.sleep(RECORD_SECONDS)
    picam2.stop_recording()


def run() -> None:
    import cv2
    from picamera2 import Picamera2  # imported here so tests never need it

    OUTPUT_DIR.mkdir(exist_ok=True)
    clip_path = OUTPUT_DIR / "clip.mp4"

    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (1280, 720)})
    picam2.configure(config)
    record_clip(picam2, clip_path)

    print(f"Recorded ~{clip_duration(RECORD_SECONDS * FPS, FPS):.1f}s clip to {clip_path}")

    cap = cv2.VideoCapture(str(clip_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices_to_sample = sample_plan(total_frames, NUM_SAMPLES)

    sampled = []
    for idx in indices_to_sample:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            sampled.append(frame)
    cap.release()

    if sampled:
        contact_sheet = np.hstack(sampled)
        out_path = OUTPUT_DIR / "clip_samples.jpg"
        cv2.imwrite(str(out_path), contact_sheet)
        print(f"Saved {len(sampled)}-frame contact sheet to {out_path}")


if __name__ == "__main__":
    run()
