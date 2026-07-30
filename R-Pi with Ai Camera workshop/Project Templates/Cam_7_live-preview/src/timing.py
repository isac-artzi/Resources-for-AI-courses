"""Pure timing/display helpers for cam-live-preview -- no hardware required."""
from __future__ import annotations


def format_elapsed(seconds: float) -> str:
    """Format a seconds count as 'M:SS' for a terminal countdown display."""
    if seconds < 0:
        raise ValueError("seconds cannot be negative")
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    return f"{minutes}:{secs:02d}"


def countdown_ticks(duration_s: int, tick_interval_s: int = 1) -> list[int]:
    """Seconds-remaining values to print during the preview, counting down
    from duration_s to 0 in steps of tick_interval_s (always including 0).
    """
    if duration_s < 0:
        raise ValueError("duration_s cannot be negative")
    if tick_interval_s <= 0:
        raise ValueError("tick_interval_s must be positive")
    ticks = list(range(duration_s, 0, -tick_interval_s))
    if not ticks or ticks[-1] != 0:
        ticks.append(0)
    return ticks
