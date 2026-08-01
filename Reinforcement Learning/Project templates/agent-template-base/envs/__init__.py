"""
envs/ — the environment, as a proper Gymnasium environment.

The base template ships no environment; each topic provides one and exposes it
through `make_env()`. The service tier imports THIS name and nothing deeper,
so a topic can swap FrozenLake for CartPole for Connect Four without touching
api/main.py.

    from envs import make_env
    env = make_env()

Two constraints that are not negotiable in this course:

  * Install only the extras you need — `gymnasium[classic-control]`, never
    `gymnasium[all]`, which still declares the obsolete swig-dependent Box2D
    package and fails to install.
  * Anything the DEPLOYED service instantiates must be classic-control.
    Box2D environments publish wheels only for CPython 3.10–3.13, have no
    source distribution, and do not fit the free-tier memory allocation.
"""

from __future__ import annotations

__all__ = ["make_env"]


def make_env(**kwargs):  # pragma: no cover - replaced by the topic layer
    raise NotImplementedError(
        "This topic's envs/__init__.py must define make_env(). "
        "See the topic README for which environment it should return."
    )
