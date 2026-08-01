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

import re

import numpy as np

# ---------------------------------------------------------------------------
# TEXT. This topic's "observation" is a string, so the train/serve skew this
# module exists to prevent now has a new and much easier way to happen.
# ---------------------------------------------------------------------------

# scikit-learn's TfidfVectorizer default token pattern, written out here rather
# than left implicit. The training tier passes this exact constant to
# `TfidfVectorizer(token_pattern=TOKEN_PATTERN, lowercase=True)`, and
# `api/reward.py` re-implements the transform against the same constant. That
# is the whole reason it is a shared constant instead of two regexes that
# happen to look alike: the day someone "improves" one of them, both move.
#
# Read it: `\b\w\w+\b` means TWO OR MORE word characters. Single-character
# tokens ("a", "I") and all punctuation are dropped. That is scikit-learn's
# default, not a choice this course made, and it matters here because it means
# the deployed head literally cannot see the difference between "not helpful"
# and "helpful" if the negation is a one-letter word. It is not, in English,
# but you should know where the blind spot is before you ship the thing.
TOKEN_PATTERN = r"(?u)\b\w\w+\b"

_TOKEN_RE = re.compile(TOKEN_PATTERN)


def tokenise(text: str) -> list[str]:
    """Lowercase, then split on TOKEN_PATTERN. The one tokeniser in this repo.

    Both tiers call this. The training tier calls it indirectly, by handing
    `TOKEN_PATTERN` to scikit-learn; the serving tier calls it directly,
    because scikit-learn is not installed there and `POST /score` still has to
    turn a string into the same vector.

    `tests/test_equivalence.py` proves the two agree end to end — text in,
    scalar reward out — rather than trusting that they do. A tokeniser
    mismatch is the text-domain version of a transposed weight matrix: the
    arithmetic still runs, the service still answers, and every score is wrong
    by an amount nothing in the system measures.
    """
    return _TOKEN_RE.findall(text.lower())


def response_length(text: str) -> int:
    """Length in TOKENS, which is the unit the length-bias regression uses.

    Characters would be a defensible alternative and gives a very similar
    correlation; tokens is chosen because it is the unit the reward head
    actually consumes, so a correlation reported in tokens is a statement
    about the model's own input rather than about its formatting.

    Whichever you pick, pick ONE. Reporting a correlation in characters and a
    threshold justified in tokens is how a passing test stops meaning anything.
    """
    return len(tokenise(text))


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
