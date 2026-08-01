"""
envs/ — the Lake Pilot environment, exposed through one name.

    from envs import make_env
    env = make_env()                            # 8x8, slippery — the product
    env = make_env("4x4", is_slippery=False)    # deterministic — the test fixture

The service tier imports `make_env` and nothing deeper, so swapping the
environment in a later topic touches this file only.

------------------------------------------------------------------------------
WHY THE SLIPPERY LAKE IS A DIFFERENT PROBLEM, NOT MERELY A HARDER ONE
------------------------------------------------------------------------------
On the deterministic lake, `step(RIGHT)` moves you right. A solution is a
*path*: a fixed sequence of moves you could write on a napkin, replay forever,
and never need to look at the world again. Search finds it; learning is
overkill.

On the slippery lake the same call moves you in the intended direction with
probability 1/3 and in each perpendicular direction with probability 1/3 each
(never backwards). Three consequences follow, and each one is a thing this
course is actually about:

  1. A path is no longer a solution. You cannot say "go right three times",
     because after the first step you no longer know where you are standing.
     The object you have to learn is a *policy* — an action for every state —
     which is why Q is a table over states rather than a list of moves.

  2. The same action from the same state returns different outcomes on
     different visits, so a single episode says almost nothing about the
     decision that produced it. Q-learning's running average over many visits
     is not an implementation detail, it is the entire response to this fact.
     It is also why a result quoted from one seed is not evidence here.

  3. The optimal policy stops being the shortest route. Standing next to a
     hole, the move with the highest expected return is often the one that
     pushes you into a wall — a wall is a no-op, a hole is terminal — so the
     agent deliberately wastes steps to keep a slip from killing it. Students
     routinely read this as a bug. It is the correct answer, and it is the
     clearest evidence you will get that the agent is maximising expected
     return rather than imitating what you would have done.

Concretely: a competent tabular agent reaches roughly 0.2-0.6 mean return on
the slippery 8x8 lake and 1.0 on the deterministic one. If your slippery number
looks like a deterministic number, check that `is_slippery` really is True.

------------------------------------------------------------------------------
Two installation constraints that are not negotiable in this course:

  * Install only the extras you need — `gymnasium[classic-control]`, never
    `gymnasium[all]`, which still declares the obsolete swig-dependent Box2D
    package and fails to install.
  * Anything the DEPLOYED service instantiates must be classic-control or
    toy-text. FrozenLake is toy-text and pure Python: `make_env()` with no
    render mode pulls in no graphics stack at all, which is precisely why the
    service tier can afford to run /rollout server-side on the free tier.
"""

from __future__ import annotations

import gymnasium as gym

__all__ = [
    "ACTION_ARROWS",
    "ACTION_NAMES",
    "ENV_ID",
    "MAP_NAME",
    "action_space_size",
    "describe",
    "lake_rows",
    "make_env",
    "observation_space_size",
]

ENV_ID = "FrozenLake-v1"
MAP_NAME = "8x8"

# Gymnasium's action order for FrozenLake. Written out here rather than looked
# up because the UI has to label arrows and because the order is part of the
# artifact's meaning: a Q-table exported under one action order and read under
# another is a policy that turns left when it meant to turn right, silently.
ACTION_NAMES = ("Left", "Down", "Right", "Up")
ACTION_ARROWS = ("←", "↓", "→", "↑")


def make_env(
    map_name: str = MAP_NAME,
    is_slippery: bool = True,
    render_mode: str | None = None,
    **kwargs,
) -> gym.Env:
    """Return the Lake Pilot environment.

    The defaults are the product's configuration — 8x8 and slippery — so that
    `make_env()` with no arguments is what the service tier rolls out and what
    every number in your README refers to. Overriding them is how the tests get
    a fast deterministic lake; it is not how you get a better score.

    `render_mode` stays None by default on purpose. `"human"` opens a pygame
    window, which on a headless host (CI, Streamlit Cloud, a Colab runtime) is
    not slow — it is a crash. The UI animates the trajectory itself from the
    state indices already present in the /rollout response, so nothing in this
    product ever needs a renderer.
    """
    return gym.make(
        ENV_ID,
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode=render_mode,
        **kwargs,
    )


def lake_rows(map_name: str = MAP_NAME) -> list[str]:
    """The map as row strings: 'S' start, 'F' frozen, 'H' hole, 'G' goal.

    Imported from Gymnasium rather than copied into this file so that the grid
    the UI paints and the grid the agent walks cannot drift apart. A hand-typed
    map with one hole in the wrong place produces a "Watch" tab where the agent
    appears to drown on solid ice — and you will go looking for the bug in the
    policy, which is the one place it is not.
    """
    from gymnasium.envs.toy_text.frozen_lake import MAPS

    return list(MAPS[map_name])


def observation_space_size(map_name: str = MAP_NAME) -> int:
    """Number of discrete states: 64 for the 8x8 lake, 16 for the 4x4."""
    n = len(lake_rows(map_name))
    return n * n


def action_space_size() -> int:
    return len(ACTION_NAMES)


def describe(map_name: str = MAP_NAME, is_slippery: bool = True) -> str:
    """The spaces, printed. Build step 3 asks you to record these in the README.

    Read the output rather than transcribing it from the documentation.
    `Discrete(64)` is the reason the artifact is a 64x4 array and the reason
    /act takes a one-element state vector holding an index — both are contracts
    that other files in this repository depend on.
    """
    env = make_env(map_name=map_name, is_slippery=is_slippery)
    max_steps = env.spec.max_episode_steps if env.spec else None
    lines = [
        f"env_id             : {ENV_ID}",
        f"map_name           : {map_name}",
        f"is_slippery        : {is_slippery}",
        f"observation_space  : {env.observation_space}",
        f"action_space       : {env.action_space}   {dict(enumerate(ACTION_NAMES))}",
        f"max_episode_steps  : {max_steps}   (a truncation, not a termination)",
        "reward             : 1.0 on reaching G, 0.0 on every other transition",
        "",
        *lake_rows(map_name),
    ]
    env.close()
    return "\n".join(lines)
