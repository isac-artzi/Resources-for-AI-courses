"""
envs/ — the environment, as a proper Gymnasium environment.

The service tier imports `make_env` from THIS name and nothing deeper, so a
later topic can swap the grid for CartPole without touching api/main.py.

    from envs import make_env
    env = make_env()

This topic ships `GridWorldEnv`, a 5x5 stochastic routing grid with an explicit
transition-probability matrix at `env.unwrapped.P`. It is an importable module
rather than a function inside a training script on purpose: Topics 3 and 5
reuse this exact environment, and an environment that lives in a notebook cell
cannot be imported, cannot be unit-tested, and cannot be pinned when a later
result disagrees with an earlier one.

Two constraints that are not negotiable in this course:

  * Install only the extras you need — `gymnasium[classic-control]`, never
    `gymnasium[all]`, which still declares the obsolete swig-dependent Box2D
    package and fails to install.
  * Anything the DEPLOYED service instantiates must be cheap. This grid is
    pure Python and NumPy and costs nothing. Box2D environments publish wheels
    only for CPython 3.10–3.13, have no source distribution, and do not fit the
    free-tier memory allocation.
"""

from __future__ import annotations

import gymnasium as gym

from envs.gridworld import (
    ACTION_ARROWS,
    ACTION_NAMES,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    GridSpec,
    GridWorldEnv,
    RewardSpec,
    make_gridworld,
)

__all__ = [
    "ACTION_ARROWS",
    "ACTION_NAMES",
    "DOWN",
    "ENV_ID",
    "LEFT",
    "RIGHT",
    "UP",
    "GridSpec",
    "GridWorldEnv",
    "RewardSpec",
    "make_env",
    "make_gridworld",
]

# The string written into `experiments.env_id`, and it carries a version. When
# you change the slip probability or move a pit you have changed the problem,
# and a results table that mixes rows from before and after that change is not
# a comparison. Bumping this suffix is how a reader tells the two apart later.
ENV_ID = "GridWorld5x5-v1"


def make_env(**kwargs) -> gym.Env:
    """The one factory both the service tier and the training tier call.

    Keyword arguments are forwarded to `make_gridworld`, so a caller that needs
    the raw object for model access can ask for `time_limit=False`. Everything
    else gets the time-limited environment — the one the numbers in the README
    were measured against.
    """
    return make_gridworld(**kwargs)
