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


# ---------------------------------------------------------------------------
# Topic 5: the encoding the learned evaluator consumes.
# ---------------------------------------------------------------------------


def canonical_planes(board, player: int) -> np.ndarray:
    """Flat board (42 cells, values -1/0/+1) -> 84 floats, side-to-move first.

    Two binary planes concatenated: cells owned by `player`, then cells owned by
    the opponent. This is the AlphaGo/AlphaZero input convention scaled down to
    a 6x7 board, and each of its three properties is a decision worth defending.

    **Why planes rather than one signed 42-vector.** A single channel forces the
    network to learn that -1 and +1 are opposites before it can learn anything
    about the game, and it makes "empty" numerically between the two players,
    which it is not. Two binary planes make ownership a linear feature.

    **Why canonical (relative to the side to move) rather than absolute.** With
    an absolute encoding the network has to learn each pattern twice, once for
    each colour, from half as much data. Canonicalising halves the function it
    has to fit, and it is the reason self-play data from both sides of a game
    can be pooled into one training set at all. The cost is that the network can
    no longer tell you "what is good for yellow" — only "what is good for
    whoever is to move" — and every consumer of its value must respect that.
    `search/net.py::value_of` is where the sign is put back.

    **Why no turn-indicator plane.** It would be constant 1 in the canonical
    encoding: the side to move is always "me". Feeding a constant feature costs
    parameters and teaches nothing. (An absolute encoding DOES need one, which
    is a good way to remember which convention you are in.)
    """
    arr = np.asarray(board, dtype=np.int8).ravel()
    if player not in (1, -1):
        raise ValueError(f"player must be +1 or -1, got {player}")
    mine = (arr == player).astype(np.float32)
    theirs = (arr == -player).astype(np.float32)
    return np.concatenate([mine, theirs])
