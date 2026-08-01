"""
envs/ — the three environments of this product, behind one factory.

Topic 4 is the first product that ships MORE THAN ONE environment, because it
ships more than one agent:

    CartPole-v1   4 obs, 2 discrete actions   — A2C, the discrete warm-up
    Acrobot-v1    6 obs, 3 discrete actions   — PPO, the harder discrete task
    Pendulum-v1   3 obs, 1 continuous action  — SAC, the continuous task

    from envs import make_env
    env = make_env("Acrobot-v1")

`make_env()` therefore takes an argument, which the base template's did not.
That one change is the reason this file exists in the shape it does: three
observation widths behind one `POST /act` contract is exactly the hazard the
required 422 test protects against, and the numbers 4, 6 and 3 have to be
written down somewhere the service tier can read them without constructing a
Gymnasium environment just to answer `/healthz`.

Why never LunarLander, even though the textbook's companion code uses it
-----------------------------------------------------------------------
Box2D — the physics engine behind LunarLander and BipedalWalker — publishes
binary wheels for CPython 3.10 through 3.13 ONLY, and ships no source
distribution. There is no fallback: on any other interpreter
`pip install gymnasium[box2d]` fails outright, and it fails at install time on
the host you do not control rather than in the terminal in front of you. It also
does not fit the free-tier memory allocation the deployed service runs inside.
Every environment this product deploys is classic-control, which is pure Python
and a few megabytes. (The textbook's code also targets `LunarLander-v2`, an id
that no longer exists in Gymnasium at all; if you adapt that code, it is `-v3`
now — and you still should not deploy it.)

Install `gymnasium[classic-control]`, never `gymnasium[all]` — the `all` extra
still declares the obsolete swig-dependent Box2D package and will fail to
install for exactly the reason above.

Note that gymnasium is imported INSIDE `make_env()` rather than at module scope.
`api/main.py` runs `from envs import make_env` at import time, so a module-scope
import would take the whole service tier down on a host where the
classic-control extra is missing, instead of failing only the endpoints that
actually need an environment.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ENV_SPECS", "DEFAULT_ENV_ID", "EnvSpec", "make_env", "spec_for"]


@dataclass(frozen=True)
class EnvSpec:
    """The shape facts about one environment, written down once.

    Read by the training tier, the export step and the service tier. All three
    must agree about how wide an observation is, and a literal `4` repeated in
    six files is precisely how they stop agreeing — usually on the evening you
    add the second environment and discover that `/act` happily accepts a
    CartPole observation for the Acrobot policy because nothing ever compared
    the two numbers.

    `random_return` is not decoration either. Every learning claim in this
    product has the form "better than doing nothing", and "nothing" has a
    number: a uniformly random policy scores about +20 on CartPole, about −500
    on Acrobot and about −1200 on Pendulum. Quoting a return without its
    baseline is the most common way a first RL report overstates a result.
    """

    env_id: str
    obs_dim: int
    discrete: bool
    n_actions: int          # number of discrete actions, or the action-vector width
    action_low: float       # continuous only; ignored when discrete
    action_high: float
    max_steps: int
    random_return: float
    threshold: float        # the "learned something" line, stated rather than assumed
    obs_labels: tuple[str, ...]


ENV_SPECS: dict[str, EnvSpec] = {
    "CartPole-v1": EnvSpec(
        env_id="CartPole-v1",
        obs_dim=4,
        discrete=True,
        n_actions=2,
        action_low=0.0,
        action_high=0.0,
        max_steps=500,          # CartPole-v1's own TimeLimit; stated, not assumed
        random_return=22.0,
        threshold=195.0,        # the classical CartPole bar, kept for continuity
        obs_labels=("cart position", "cart velocity", "pole angle", "pole angular velocity"),
    ),
    "Acrobot-v1": EnvSpec(
        env_id="Acrobot-v1",
        obs_dim=6,
        discrete=True,
        n_actions=3,
        action_low=0.0,
        action_high=0.0,
        max_steps=500,
        # −500 is a FLOOR, not the mean of a distribution around it: the reward
        # is −1 per step until the goal height is reached, a random policy
        # essentially never reaches it, and so almost every random episode
        # truncates at exactly −500. An agent scoring −450 has genuinely learned
        # something, yet the two numbers are only 10% apart on the axis. Say so
        # next to the chart, or your reader will misjudge the size of the effect.
        random_return=-500.0,
        threshold=-100.0,
        obs_labels=("cos θ1", "sin θ1", "cos θ2", "sin θ2", "ω1", "ω2"),
    ),
    "Pendulum-v1": EnvSpec(
        env_id="Pendulum-v1",
        obs_dim=3,
        discrete=False,
        n_actions=1,            # one continuous torque
        # The action range matters and is the thing most often dropped. A tanh
        # squash produces [−1, 1]; Pendulum's torque lives in [−2, 2]. Deploying
        # the unscaled tanh output gives an agent that can only ever apply half
        # the torque it trained with, which reads as a policy that "almost"
        # works. The scale travels inside the artifact — see
        # train/export.export_squashed_gaussian — so the serving path cannot
        # forget it.
        action_low=-2.0,
        action_high=2.0,
        max_steps=200,
        random_return=-1200.0,
        threshold=-300.0,
        obs_labels=("cos θ", "sin θ", "angular velocity"),
    ),
}

# What `make_env()` returns when called with no argument. The base template's
# `/rollout` handler calls `make_env()` bare, so keeping a default means a fresh
# clone answers before you have wired the env id through the artifact metadata.
DEFAULT_ENV_ID = "CartPole-v1"


def spec_for(env_id: str) -> EnvSpec:
    """Look up an environment's shape facts, or fail with the list of legal ids.

    A bare `KeyError: 'Pendulum-v0'` tells the reader nothing. This tells them
    what they may have meant, which matters because two of the three ids in this
    product changed suffix in living memory and the textbook still uses an old
    one.
    """
    try:
        return ENV_SPECS[env_id]
    except KeyError:
        raise KeyError(
            f"unknown env_id {env_id!r}. This product ships exactly three: "
            f"{', '.join(sorted(ENV_SPECS))}. LunarLander is deliberately absent — "
            "see the note at the top of envs/__init__.py."
        ) from None


def make_env(env_id: str | None = None, render_mode: str | None = None, **kwargs):
    """Return a FRESH environment for `env_id`.

    Fresh on every call, deliberately: a Gymnasium environment carries a step
    counter and RNG state, and sharing one instance between the training loop
    and a server-side rollout is a reproducibility bug that surfaces later as
    "my evaluation numbers moved and I did not change anything".

    Remember that `step()` returns a FIVE-tuple
    `(obs, reward, terminated, truncated, info)`. Code written against the old
    four-tuple Gym API will not run, and the way it fails — an unpacking error
    deep inside a training loop — reads like a bug in your code rather than the
    API change it actually is.

    The `terminated` / `truncated` split is not cosmetic in this topic. A2C, PPO
    and SAC all bootstrap the value of the final state, and the correct
    bootstrap differs between the two cases: a TRUNCATED episode has a future
    worth V(s'), a TERMINATED one has a future worth exactly zero. Collapsing
    them into a single `done` flag — which every pre-2021 tutorial does — teaches
    the critic that surviving to CartPole's 500-step limit is as bad as dropping
    the pole, and the learning curve then plateaus for a reason you will never
    find by tuning the learning rate.
    """
    env_id = env_id or DEFAULT_ENV_ID
    spec_for(env_id)  # validate first, so a typo produces the readable error above

    try:
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "gymnasium is not installed here. The service tier needs it because "
            "POST /rollout runs episodes server-side; install "
            "`gymnasium[classic-control]` — never `gymnasium[all]`."
        ) from exc

    return gym.make(env_id, render_mode=render_mode, **kwargs)
