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
and why, is part of the README — there is a table there, and this module is the
code it describes.

The two learners in this product make the point concretely:

    value iteration  needs  dense_model()          — P as arrays
                     needs  NOTHING else: it never sees an observation, so
                            one-hot encoding it would be encoding a thing that
                            does not exist in its input.

    Monte Carlo      needs  discounted_returns()   — a reward SEQUENCE folded
                            backwards into a return target, which is the only
                            transformation in this file that cannot be applied
                            one sample at a time
                     needs  first_visit_indices()  — which of the repeated
                            visits inside one episode is allowed to contribute
                     needs  NO encoding either, because it is tabular; the
                            moment you replace the table with a network,
                            one_hot() is what you reach for.

Every function here is unit-tested in tests/test_preprocess.py, and the pairs
that are supposed to invert each other are tested as round trips. A transform
whose inverse you have never run is a transform you do not actually know the
convention of.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

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


def from_one_hot(vector: np.ndarray) -> int:
    """Indicator vector -> the discrete state index. The inverse of one_hot().

    It exists so that the round trip can be TESTED. An encoder without a
    decoder is a place where an off-by-one hides indefinitely: the learner
    trains happily on a shifted encoding and only the deployed agent, which
    decodes differently, behaves oddly.

    Rejects a vector that is not a single indicator rather than silently
    argmaxing it — a near-one-hot vector arriving here almost always means a
    softmax output was passed by mistake, and returning its argmax would turn
    that mistake into a plausible-looking wrong answer.
    """
    v = np.asarray(vector)
    if v.ndim != 1 or v.size == 0:
        raise ValueError(f"expected a non-empty 1-D vector, got shape {v.shape}")
    hits = np.flatnonzero(np.isclose(v, 1.0))
    if hits.size != 1 or not np.isclose(v.sum(), 1.0):
        raise ValueError("not a one-hot vector: exactly one entry must be 1 and the rest 0")
    return int(hits[0])


def state_index(row: int, col: int, n_cols: int) -> int:
    """(row, col) -> the flat state index used by the environment and the tables.

    Row-major, and that choice is load-bearing rather than arbitrary: every
    value function in this repository is stored as a flat length-25 array and
    reshaped to (5, 5) for the heat maps. Encode column-major here and the heat
    map is a transposed picture of a correct value function, which looks
    plausible enough to ship.
    """
    if n_cols < 1:
        raise ValueError("n_cols must be >= 1")
    if row < 0 or col < 0 or col >= n_cols:
        raise ValueError(f"({row}, {col}) is not a valid cell in a grid of width {n_cols}")
    return row * n_cols + col


def index_to_state(index: int, n_cols: int) -> tuple[int, int]:
    """Flat index -> (row, col). The inverse of state_index()."""
    if n_cols < 1:
        raise ValueError("n_cols must be >= 1")
    if index < 0:
        raise ValueError(f"state index {index} must be non-negative")
    return divmod(int(index), n_cols)


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


def bin_centre(index: int, low: float, high: float, bins: int) -> float:
    """Bin index -> the representative value of that bin. The partner of discretise().

    Discretisation is lossy, so this is not a true inverse and the test asserts
    the honest property instead: the recovered value is within half a bin width
    of the original. That half-width IS the resolution you gave up, and it is
    the number to quote when someone asks what your discretisation cost. If it
    is larger than the differences your policy is supposed to react to, the
    discretisation has thrown away the problem.
    """
    if bins < 1:
        raise ValueError("bins must be >= 1")
    if high <= low:
        raise ValueError("high must exceed low")
    if not 0 <= index < bins:
        raise ValueError(f"bin index {index} outside [0, {bins})")
    width = (high - low) / bins
    return float(low + (index + 0.5) * width)


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
# The two transformations a Monte Carlo learner cannot do without.
# ---------------------------------------------------------------------------


def discounted_returns(rewards: Sequence[float], gamma: float) -> np.ndarray:
    """Reward sequence -> the return G_t at every time step of one episode.

        G_t = r_{t+1} + gamma * G_{t+1},   G_T = 0

    This is the transformation that separates Monte Carlo from every
    single-sample method in this repository: the learning target for step t is
    a function of the whole REMAINDER of the episode, so it cannot be computed
    until the episode has ended. That is the algorithmic fact behind "Monte
    Carlo methods only apply to episodic tasks" — with no terminal state there
    is no remainder to fold.

    Computed backwards in one pass rather than as a sum over powers of gamma at
    each t. The forward version is O(T^2) and, worse, accumulates gamma**t,
    which underflows to exactly 0 for t past a few hundred at gamma = 0.9 and
    silently truncates long returns.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    out = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = float(rewards[t]) + gamma * running
        out[t] = running
    return out


def first_visit_indices(states: Iterable[int]) -> dict[int, int]:
    """State sequence -> {state: index of its FIRST occurrence}.

    First-visit Monte Carlo averages one return per state per episode: the one
    from the first time the state was reached. Every-visit averages all of
    them. Both converge to v_pi, but only the first-visit returns are
    independent across the samples that make up one state's average — the
    second visit's return is a suffix of the first visit's, so every-visit
    samples inside an episode are correlated and the usual standard-error
    formula understates the uncertainty.

    That is precisely why this function returns a MAP rather than a filtered
    list: `train/monte_carlo.py` uses it to decide which time steps count, and
    the every-visit variant simply does not call it. Making the difference one
    line of code in one place is what lets the Concepts tab claim the two
    estimators differ only in that respect.
    """
    first: dict[int, int] = {}
    for t, s in enumerate(states):
        first.setdefault(int(s), t)
    return first


# ---------------------------------------------------------------------------
# The transformation a PLANNER needs — and the only one it needs.
# ---------------------------------------------------------------------------


def dense_model(
    P: Mapping[int, Mapping[int, Sequence[tuple[float, int, float, Any]]]],
    n_states: int,
    n_actions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gymnasium's nested-dict model -> (T, R, bootstrap) arrays.

        T[s, a, s'] = P(s' | s, a)
        R[s, a]     = E[r | s, a]                       (expected over s')
        B[s, a, s'] = 1 if s' may be bootstrapped from, 0 if the episode ended

    This is preprocessing in exactly the sense the rest of this module means
    it: a transformation between raw environment output and learner input. It
    just happens that the planner's "input" is the model rather than an
    observation, which is the cleanest illustration available of why one
    preprocessing pipeline for all algorithms is a fiction.

    Two details that are worth more than they look:

      * A whole sweep of value iteration becomes `(R + gamma * (T * B) @ V)`,
        one einsum over 25 x 4 x 25 numbers instead of 100 Python loops. On
        this grid either is instant; on the 10^4-state problems this same code
        is expected to handle in Topic 5, the loop version is the difference
        between a coffee and an afternoon.
      * B is separate from T because a terminal successor must contribute its
        reward but NOT its value. Folding "terminated" into the transition
        probabilities instead — the shortcut everyone tries first — makes the
        rows stop summing to one, and the resulting value function is wrong in
        a way that still looks smooth on a heat map.
    """
    T = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    R = np.zeros((n_states, n_actions), dtype=np.float64)
    B = np.ones((n_states, n_actions, n_states), dtype=np.float64)
    for s in range(n_states):
        for a in range(n_actions):
            for prob, next_state, reward, terminated in P[s][a]:
                T[s, a, next_state] += prob
                R[s, a] += prob * reward
                if terminated:
                    B[s, a, next_state] = 0.0

    row_sums = T.sum(axis=2)
    if not np.allclose(row_sums, 1.0, atol=1e-9):
        bad = np.argwhere(~np.isclose(row_sums, 1.0, atol=1e-9))[:5]
        raise ValueError(
            "transition rows must sum to 1; offending (state, action) pairs: "
            f"{[tuple(int(v) for v in pair) for pair in bad]}. A row that sums "
            "to less than 1 usually means a terminal state was left out of P."
        )
    return T, R, B
