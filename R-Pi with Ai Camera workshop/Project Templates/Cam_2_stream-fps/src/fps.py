"""Pure timing math for cam-stream-fps -- no camera hardware required."""
from __future__ import annotations


def compute_fps(timestamps: list[float]) -> float:
    """Average FPS implied by a list of frame-arrival timestamps (seconds).

    Needs at least 2 timestamps (1 interval) to compute anything.
    """
    if len(timestamps) < 2:
        raise ValueError("need at least 2 timestamps to compute an FPS")
    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0:
        raise ValueError("timestamps must be strictly increasing")
    intervals = len(timestamps) - 1
    return intervals / elapsed


def summarize_runs(runs: dict[str, list[float]]) -> dict[str, dict]:
    """Given {label: [timestamps]} (e.g. one entry per resolution tested),
    return {label: {"fps": ..., "frame_count": ..., "elapsed_s": ...}}.
    """
    summary = {}
    for label, timestamps in runs.items():
        fps = compute_fps(timestamps)
        summary[label] = {
            "fps": round(fps, 2),
            "frame_count": len(timestamps),
            "elapsed_s": round(timestamps[-1] - timestamps[0], 3),
        }
    return summary


def format_summary(summary: dict[str, dict]) -> str:
    """Render a summarize_runs() result as an aligned text table."""
    if not summary:
        return "(no runs recorded)"
    label_width = max(len(label) for label in summary)
    lines = [f"{'Resolution'.ljust(label_width)}   FPS     Frames   Elapsed"]
    for label, stats in summary.items():
        lines.append(
            f"{label.ljust(label_width)}   {stats['fps']:<6}  "
            f"{stats['frame_count']:<7}  {stats['elapsed_s']}s"
        )
    return "\n".join(lines)
