"""
envs/gym_env.py — the Gymnasium wrapper around the rules in connect_four.py.

Why this is a SEPARATE module from `envs/connect_four.py`, and why `make_env()`
imports it lazily:

`gymnasium` is a serving requirement in this course because `POST /rollout` runs
evaluation episodes server-side. But nothing else in the deployed process needs
it: the search agents, the learned evaluator and `POST /act` all work on
`Position` objects and never construct an environment. Keeping the wrapper in
its own module means the import graph reflects that — `import api.main` loads
the rules and the search, and does NOT load gymnasium, its space classes, its
wrapper machinery or its registry.

That is worth about 5 MB of resident memory and, more importantly, it is a
checkable claim rather than a hopeful one: CI uninstalls gymnasium and imports
`search.agents` to prove it. Before this split the claim was in a comment in
requirements-serve.txt and was quietly false, because
`envs/connect_four.py` had `import gymnasium` at module scope inside a
try/except. A defensive import still imports.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.connect_four import (
    COLS,
    STATE_DIM,
    IllegalMoveError,
    Position,
    encode_state,
)


class ConnectFourEnv(gym.Env):
    """Connect Four as an ALTERNATING-MOVE Gymnasium environment.

    Gymnasium's contract is written for one agent against an environment, and
    Connect Four has two agents. There are two honest ways to resolve that:

      (a) fix an opponent inside the environment, so `step()` plays the agent's
          move and the opponent's reply. The observation is then always from the
          learner's side and the standard single-agent machinery applies — but
          the environment now contains a policy, and every measurement is
          against that particular opponent.
      (b) alternate: `step()` plays ONE move, and the observation carries whose
          turn it is. The caller drives both sides.

    This environment does (b), because this product's whole subject is agents
    playing each other and (a) would bake one of them into the environment. The
    consequence, which you must keep in mind, is that `reward` is returned from
    the point of view of the player who JUST MOVED, not of the player about to
    move. A reward of +1 means "the move you just played won"; the next
    observation belongs to the opponent, who has just lost.

    If you want the single-agent shape for a learner, wrap this rather than
    change it — `train/selfplay.py` drives `Position` directly, which is the
    other way to do it.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        # Box rather than MultiDiscrete: the observation crosses the API as a
        # list of floats (see `encode_state`), and declaring it as floats here
        # means the space and the wire format cannot drift apart.
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(COLS)
        self.render_mode = render_mode
        self.position = Position()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.position = Position()
        return self._obs(), self._info()

    def step(self, action: int):
        mover = self.position.player
        try:
            self.position.push(int(action))
        except IllegalMoveError:
            # An illegal move ENDS THE GAME as a loss for the mover rather than
            # raising. Raising is right for the search agents (they read the
            # mask, and a violation is a bug); it is wrong for a learner, which
            # would otherwise crash the training loop the first time an
            # untrained network put mass on a full column. Losing is a gradient;
            # a traceback is not.
            return self._obs(), -1.0, True, False, self._info(illegal=True)

        terminated = self.position.is_terminal()
        reward = self.position.result_for(mover) if terminated else 0.0
        return self._obs(), float(reward), terminated, False, self._info()

    def render(self):  # pragma: no cover - convenience only
        return str(self.position)

    def _obs(self) -> np.ndarray:
        return np.asarray(encode_state(self.position), dtype=np.float32)

    def _info(self, illegal: bool = False) -> dict[str, Any]:
        return {
            "action_mask": self.position.action_mask(),
            "player": self.position.player,
            "winner": self.position.winner,
            "illegal_move": illegal,
        }
