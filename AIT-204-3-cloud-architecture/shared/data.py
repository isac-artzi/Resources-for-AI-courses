"""Synthetic dataset generator.

Lives in ``shared`` so that BOTH the API tier (to train) and the db/seed script
(to persist a default dataset) import the exact same generating function. This
guarantees the "known ground truth" that the numerical pytest test checks
against.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def generate_linear(
    slope: float,
    intercept: float,
    noise: float,
    n_points: int,
    x_low: float = -10.0,
    x_high: float = 10.0,
    seed: int = 42,
) -> Tuple[List[float], List[float]]:
    """Return ``(xs, ys)`` sampled from ``y = slope*x + intercept + N(0, noise)``.

    A fixed ``seed`` makes the dataset reproducible so run comparisons in the
    Supabase ``runs`` table are apples-to-apples.
    """
    rng = np.random.default_rng(seed)
    xs = rng.uniform(x_low, x_high, size=n_points)
    ys = slope * xs + intercept + rng.normal(0.0, noise, size=n_points)
    return xs.tolist(), ys.tolist()
