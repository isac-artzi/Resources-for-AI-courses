"""
envs/ — the environment, as a proper Gymnasium-style environment.

This topic's product is a SCORER, so the service tier's `POST /rollout` is not
where the interesting work happens. It still needs an environment, and the one
exposed here is chosen to make a point rather than to fill a slot:

    the iterated prisoner's dilemma AGAINST A FIXED OPPONENT.

That is the "frozen agent" arm of `train/multiagent.py`'s non-stationarity
experiment, and it is a genuine environment precisely BECAUSE the opponent is
frozen. Freeze the opponent and the transition function is fixed, the Markov
property holds, and every algorithm from Topics 1 to 5 applies unchanged. Let
the opponent learn and none of that is true any more — there is no `Env` you
could write, because `step` would have to depend on training time.

So: `envs/make_env()` exists, and the fact that it CAN exist is the lesson.
The multi-agent experiments live in `train/multiagent.py` and deliberately do
not go through this interface, because they cannot.

    from envs import make_env
    env = make_env(opponent="tit_for_tat")

Two constraints that are not negotiable in this course:

  * Install only the extras you need — `gymnasium[classic-control]`, never
    `gymnasium[all]`, which still declares the obsolete swig-dependent Box2D
    package and fails to install.
  * Anything the DEPLOYED service instantiates must be cheap. This environment
    is pure Python arithmetic over a five-element table, which is the standard
    to hold a serving-tier environment to.
"""

from __future__ import annotations

import numpy as np

__all__ = ["make_env", "IPDVersusFixedOpponent", "OPPONENTS", "PAYOFF"]

# Payoffs, defined here by value rather than imported from train/ — the serving
# tier must never import the training tier, and a four-entry table is not worth
# breaking that rule for. If you change them, change them in both places in the
# same commit; `tests/test_multiagent.py` asserts the two copies agree, so the
# duplication is checked rather than merely hoped for.
PAYOFF = {(0, 0): (3.0, 3.0), (0, 1): (0.0, 5.0), (1, 0): (5.0, 0.0), (1, 1): (1.0, 1.0)}

COOPERATE, DEFECT = 0, 1

# Fixed opponent policies, indexed by the memory-1 state
#   0 = (C,C)   1 = (C,D)   2 = (D,C)   3 = (D,D)   4 = start
# where the FIRST element of the pair is the learner's previous action.
OPPONENTS: dict[str, list[int]] = {
    # Copy the learner's last move. Opens with cooperation.
    "tit_for_tat": [COOPERATE, COOPERATE, DEFECT, DEFECT, COOPERATE],
    # Never cooperates. The best response is to defect, and an agent that does
    # not find that is broken in a way worth catching early.
    "always_defect": [DEFECT] * 5,
    # Always cooperates. The best response is to ALWAYS DEFECT, for 5.0 a step.
    # An agent that learns to cooperate against this one has learned to be nice
    # rather than to maximise, which is a useful thing to be able to show.
    "always_cooperate": [COOPERATE] * 5,
    # Cooperates until the learner defects once, then defects forever.
    # The observation is memory-1, so this policy is NOT a function of what the
    # agent can see — it is implemented with an internal flag. Included on
    # purpose: it is the cheapest demonstration of partial observability in this
    # repository, and an agent that keeps trying to defect "just once" against
    # it is showing you what a hidden state costs.
    "grim": [],
}


class IPDVersusFixedOpponent:
    """Single-agent iterated prisoner's dilemma against a fixed opponent.

    Observation: the memory-1 state index, 0..4 (see OPPONENTS above).
    Action:      0 = cooperate, 1 = defect.
    Reward:      the learner's payoff from the stage game.

    Implements the three methods `api/main.py::rollout` calls — `reset`,
    `step`, `close` — with Gymnasium's return signatures, rather than
    subclassing `gymnasium.Env`. Subclassing would pull Gymnasium into the
    serving import graph for the sake of two `spaces` objects nothing here
    reads, and the discipline of this course is not paying for imports you do
    not use.
    """

    n_states = 5
    n_actions = 2

    def __init__(self, opponent: str = "tit_for_tat", max_steps: int = 50) -> None:
        if opponent not in OPPONENTS:
            raise ValueError(f"unknown opponent {opponent!r}; choose from {sorted(OPPONENTS)}")
        self.opponent = opponent
        self.max_steps = max_steps
        self._rng = np.random.default_rng(0)
        self.reset()

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._last: tuple[int, int] | None = None
        self._steps = 0
        self._grim_triggered = False
        return 4, {}          # state 4 is "no history yet"

    def _opponent_action(self, state: int) -> int:
        if self.opponent == "grim":
            return DEFECT if self._grim_triggered else COOPERATE
        return OPPONENTS[self.opponent][state]

    def step(self, action: int):
        action = int(action)
        if action not in (COOPERATE, DEFECT):
            raise ValueError(f"action must be 0 (cooperate) or 1 (defect), got {action}")
        state = 4 if self._last is None else self._last[0] * 2 + self._last[1]
        opp = self._opponent_action(state)
        reward, _ = PAYOFF[(action, opp)]
        if action == DEFECT:
            self._grim_triggered = True
        self._last = (action, opp)
        self._steps += 1
        next_state = self._last[0] * 2 + self._last[1]
        # TRUNCATED, not terminated. The IPD has no terminal state — the episode
        # ends because we stopped playing, not because the game did. Reporting
        # it as termination would tell a bootstrapping learner that the value of
        # the final state is zero, which it is not, and would bias every value
        # estimate near the horizon downward.
        return next_state, reward, False, self._steps >= self.max_steps, {}

    def close(self) -> None:
        return None


def make_env(opponent: str = "tit_for_tat", max_steps: int = 50, **kwargs):
    del kwargs      # accepted and ignored, so callers may pass render_mode etc.
    return IPDVersusFixedOpponent(opponent=opponent, max_steps=max_steps)
