"""Pure clip-timing helpers for cam-clip-record-playback -- no hardware required."""
from __future__ import annotations


def clip_duration(frame_count: int, fps: float) -> float:
    """Seconds of footage represented by frame_count frames at fps."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if frame_count < 0:
        raise ValueError("frame_count cannot be negative")
    return frame_count / fps


def frame_index_at_time(fps: float, seconds: float) -> int:
    """Which frame index (0-based) corresponds to a given timestamp."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if seconds < 0:
        raise ValueError("seconds cannot be negative")
    return int(seconds * fps)


def sample_plan(total_frames: int, num_samples: int) -> list[int]:
    """Evenly spaced frame indices to sample across a clip, e.g. for
    building a contact sheet of num_samples frames from a longer clip.
    Always includes frame 0 and the last frame when num_samples >= 2.
    """
    if total_frames < 1:
        raise ValueError("total_frames must be >= 1")
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    if num_samples == 1:
        return [0]
    if num_samples > total_frames:
        num_samples = total_frames
    last = total_frames - 1
    return sorted({round(last * i / (num_samples - 1)) for i in range(num_samples)})
