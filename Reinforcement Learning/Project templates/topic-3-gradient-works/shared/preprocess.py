"""
shared/preprocess.py — every transformation between raw environment output and
learner input lives here, and nowhere else.

The rule is enforced socially rather than mechanically: no `train/` module may
reshape, clip, normalise or encode inline. If it needs a transformation, it
imports it from this file. The reason is that a preprocessing step applied
during training and forgotten at serving time is the single most common cause
of a policy that scores well offline and behaves randomly in the deployed app.
Keeping the functions in one importable place makes that failure a diff rather
than a mystery.

Note also that DIFFERENT ALGORITHMS NEED DIFFERENT PREPROCESSING, and that is
a design decision to document rather than a nuisance to hide. A value-iteration
planner consumes the transition model directly and needs no encoding at all.
A tabular Monte Carlo learner needs a state index. A neural policy needs a
float vector, usually normalised. Saying which of these your learner requires,
and why, is part of the README.
"""

from __future__ import annotations

import numpy as np


def one_hot(index: int, n: int) -> np.ndarray:
    """Discrete state index -> a length-n indicator vector.

    Used when a *function approximator* consumes a discrete state. A tabular
    learner does not need this; it indexes the table directly.
    """
    if not 0 <= index < n:
        raise ValueError(f"state index {index} outside [0, {n})")
    v = np.zeros(n, dtype=np.float32)
    v[index] = 1.0
    return v


def discretise(value: float, low: float, high: float, bins: int) -> int:
    """Continuous scalar -> a bin index, clipped at both ends.

    Clipping rather than raising is deliberate: an observation outside the
    range you assumed is a normal event at deployment time, and a crash is a
    worse answer than the nearest bin. Log how often it happens.
    """
    if bins < 1:
        raise ValueError("bins must be >= 1")
    span = high - low
    if span <= 0:
        raise ValueError("high must exceed low")
    scaled = (value - low) / span
    return int(np.clip(int(scaled * bins), 0, bins - 1))


def clip_reward(r: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Bound the reward magnitude.

    This lowers gradient variance and is why so many published results use it.
    It also CHANGES THE PROBLEM: an agent that cannot see the difference
    between a reward of 5 and a reward of 50 will not prefer the second. Clip
    if you must, then say in the README that you did.
    """
    return float(np.clip(r, lo, hi))


def normalise_returns(returns: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Zero-mean, unit-variance returns within a batch.

    Standard for policy gradients: it makes the update scale-free, so the same
    learning rate works across reward scales. Note that it is a *batch*
    statistic — with a batch of one it is meaningless, and with a small batch
    it is noisy enough to be its own source of instability.
    """
    returns = np.asarray(returns, dtype=np.float64)
    if returns.size < 2:
        return np.zeros_like(returns)
    return (returns - returns.mean()) / (returns.std() + eps)


def normalise_observation(obs: np.ndarray, mean: np.ndarray, std: np.ndarray,
                          eps: float = 1e-8) -> np.ndarray:
    """Standardise an observation with statistics computed at TRAINING time.

    The mean and std must be exported alongside the weights and applied
    identically at serving time. Recomputing them from live traffic is a
    classic train/serve skew bug.
    """
    return ((np.asarray(obs, dtype=np.float64) - mean) / (std + eps)).astype(np.float32)
