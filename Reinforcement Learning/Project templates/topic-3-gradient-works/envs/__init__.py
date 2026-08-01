"""
envs/ — the environment, as a proper Gymnasium environment.

Topic 3 uses **CartPole-v1** from `gymnasium[classic-control]`: four continuous
observations (cart position, cart velocity, pole angle, pole angular velocity),
two discrete actions (push left, push right), +1 reward per surviving step, and
a 500-step time limit. It is the standard first benchmark for a policy gradient
because the optimum is reachable in a few hundred episodes on a laptop, and the
failure modes this product exists to measure — high-variance gradients and
premature determinism — show up in the learning curve rather than behind a week
of compute.

    from envs import make_env
    env = make_env()

Two constraints that are not negotiable in this course:

  * Install only the extras you need — `gymnasium[classic-control]`, never
    `gymnasium[all]`, which still declares the obsolete swig-dependent Box2D
    package and fails to install.
  * Anything the DEPLOYED service instantiates must be classic-control.
    Box2D environments publish wheels only for CPython 3.10–3.13, have no
    source distribution, and do not fit the free-tier memory allocation.

Note that gymnasium is imported INSIDE `make_env()` rather than at module
scope. `api/main.py` runs `from envs import make_env` at import time, so a
module-scope import would take the whole service tier down on a host where the
classic-control extra is missing, instead of failing only the one endpoint that
actually needs an environment.
"""

from __future__ import annotations

__all__ = ["ENV_ID", "OBS_DIM", "N_ACTIONS", "make_env"]

# Named once, imported everywhere. The training tier, the export step and the
# `experiments.env_id` column must all agree, and a literal string repeated in
# four files is exactly how they stop agreeing.
ENV_ID = "CartPole-v1"

# The observation and action shapes are constants of this environment, and the
# artifact registry records them so /act can reject a 3-vector with a readable
# 422 instead of a 500. You could read them from a live env instead — but doing
# that at import time would make the service tier construct a Gymnasium
# environment just to answer /healthz.
OBS_DIM = 4
N_ACTIONS = 2


def make_env(render_mode: str | None = None, **kwargs):
    """Return a FRESH CartPole-v1 environment.

    Fresh on every call, deliberately: a Gymnasium environment carries a step
    counter and RNG state, and sharing one instance between the training loop
    and a server-side rollout is a reproducibility bug that surfaces later as
    "my evaluation numbers moved and I did not change anything".

    Remember that `step()` returns a FIVE-tuple
    `(obs, reward, terminated, truncated, info)`. Code written against the old
    four-tuple Gym API will not run, and the way it fails — an unpacking error
    deep inside a training loop — reads like a bug in your code rather than the
    API change it actually is.
    """
    try:
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "gymnasium is not installed here. The service tier needs it because "
            "POST /rollout runs episodes server-side; install "
            "`gymnasium[classic-control]` — never `gymnasium[all]`."
        ) from exc

    return gym.make(ENV_ID, render_mode=render_mode, **kwargs)
