"""
envs/gridworld.py — the 5x5 routing grid, as a proper Gymnasium environment
with an EXPLICIT transition-probability matrix.

Why an explicit matrix and not just a `step()` function
-------------------------------------------------------
This product's whole argument is planning versus learning on one problem. The
planner is only allowed to exist because the model is written down: value
iteration reads `env.unwrapped.P` and never calls `step()`. The Monte Carlo
learner is only honest because it does the opposite: it calls `step()` and is
never handed `P`. If the model lived only inside `step()`, the planner would
have to re-derive it and the comparison would be between two implementations
rather than between two ideas.

So `P` is built once, up front, in Gymnasium's toy-text convention:

    P[state][action] -> [(probability, next_state, reward, terminated), ...]

That is the same layout `gymnasium.envs.toy_text.FrozenLakeEnv` exposes, which
means every planner you write against this environment also runs against
FrozenLake with no changes — and it is why `train/value_iteration.py` contains
no environment-specific code at all.

Read `P` through `env.unwrapped.P`, never `env.P`
--------------------------------------------------
`make_env()` returns the environment inside a `TimeLimit` wrapper. Wrappers
forward `step`/`reset` but do NOT forward arbitrary attributes in Gymnasium
1.x, so `env.P` raises `AttributeError` while `env.unwrapped.P` works. This
trips up nearly everyone once. Getting into the habit of `.unwrapped` for
model access is cheaper than debugging it later under a deadline.

Layout of the default grid (row 0 at the top):

        col:  0    1    2    3    4
    row 0     S    .    .    .    .
    row 1     .    .    .    X    .
    row 2     .    .    .    .    .
    row 3     .    X    .    .    .
    row 4     .    .    .    .    G

    S = start   G = goal (+1, terminal)   X = pit (-1, terminal)

Every cell also charges a small step cost, so "arrive soon" and "arrive alive"
are both in the objective and the optimal policy has to trade them off. With a
zero step cost the optimal policy is wildly under-determined — dozens of routes
tie — and a value-iteration test that asserts an exact policy becomes flaky for
reasons that have nothing to do with the student's code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Action encoding. Fixed here and imported everywhere else, because a policy
# artifact is a bare array of action indices: if the UI and the trainer
# disagree about which index means "right", nothing errors — the agent simply
# walks into walls and you spend an evening blaming the learning rate.
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_NAMES = ("up", "right", "down", "left")
ACTION_ARROWS = ("^", ">", "v", "<")

# (d_row, d_col) for each action index above.
_DELTAS: tuple[tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass(frozen=True)
class RewardSpec:
    """The configurable reward function, as data rather than as code.

    A dataclass rather than four keyword arguments because these four numbers
    ARE the specification of the task: change `pit_penalty` from -1 to -20 and
    you have not tuned a hyperparameter, you have asked for a different policy.
    Keeping them in one named object means an experiment row can carry the
    whole reward specification in its `hyperparameters` jsonb and a reviewer
    can see exactly which problem was solved.

    If you need a reward that is not expressible here — one that depends on the
    action taken, or on a step counter — pass `reward_fn` to the environment
    instead. This spec is the default, not the limit.
    """

    step_cost: float = -0.02
    goal_reward: float = 1.0
    pit_penalty: float = -1.0

    def as_dict(self) -> dict[str, float]:
        return {
            "step_cost": self.step_cost,
            "goal_reward": self.goal_reward,
            "pit_penalty": self.pit_penalty,
        }


@dataclass
class GridSpec:
    """Geometry and terminal cells, in (row, col) coordinates."""

    rows: int = 5
    cols: int = 5
    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] = (4, 4)
    pits: tuple[tuple[int, int], ...] = ((1, 3), (3, 1))
    # Probability that the wind pushes the agent perpendicular to the action it
    # chose. NOT decoration: with slip = 0 the environment is deterministic,
    # every sampled return equals the true value exactly, and a Monte Carlo
    # estimator converges after a single visit per state. The convergence study
    # this product is built around would then measure nothing at all.
    slip: float = 0.2
    max_episode_steps: int = 100
    reward: RewardSpec = field(default_factory=RewardSpec)


class GridWorldEnv(gym.Env):
    """A 5x5 stochastic routing grid with a tabulated transition model.

    Spaces
        observation : Discrete(rows * cols), the flat index row * cols + col
        action      : Discrete(4), see ACTION_NAMES

    Attributes
        P : {state: {action: [(prob, next_state, reward, terminated), ...]}}
        terminal_states : frozenset[int]

    The observation is a plain `int`, matching Gymnasium's toy-text
    environments. The service tier lifts it into `[float(index)]` at the API
    boundary (see `api/main.py::_as_vector`) because `POST /act` carries one
    observation shape for all six products in this course. Do not one-hot
    encode it here: encoding is a LEARNER's requirement, it differs per
    algorithm, and it therefore belongs in `shared/preprocess.py`.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        spec: GridSpec | None = None,
        reward_fn: Callable[[int, int, int], float] | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.spec_ = spec or GridSpec()
        self.render_mode = render_mode

        self.rows, self.cols = self.spec_.rows, self.spec_.cols
        self.n_states = self.rows * self.cols
        self.n_actions = 4

        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(self.n_actions)

        self.start_state = self.to_index(*self.spec_.start)
        self.goal_state = self.to_index(*self.spec_.goal)
        self.pit_states = frozenset(self.to_index(r, c) for r, c in self.spec_.pits)
        self.terminal_states = frozenset({self.goal_state}) | self.pit_states

        # R(s, a, s'): reward for ARRIVING in s'. The alternative convention,
        # R(s, a), cannot express "you are paid for reaching the goal, whatever
        # you were aiming at when the wind blew you there" — and under a
        # stochastic transition that difference is the whole problem. Say which
        # convention you used in the model card; papers differ.
        self._reward_fn = reward_fn or self._default_reward

        self.P: dict[int, dict[int, list[tuple[float, int, float, bool]]]] = self._build_model()

        # Flattened copy of P for fast sampling in step(). Built once because
        # a Monte Carlo run in this product executes a few million steps, and
        # rebuilding a probability vector per step turns a 20-second job into a
        # 4-minute one.
        self._cum, self._next, self._rew, self._done = self._flatten_model()

        self.state: int = self.start_state
        self._elapsed = 0

    # -- coordinates ---------------------------------------------------------

    def to_index(self, row: int, col: int) -> int:
        return row * self.cols + col

    def to_rowcol(self, state: int) -> tuple[int, int]:
        return divmod(int(state), self.cols)

    # -- the model -----------------------------------------------------------

    def _default_reward(self, state: int, action: int, next_state: int) -> float:
        if next_state == self.goal_state:
            return self.spec_.reward.goal_reward
        if next_state in self.pit_states:
            return self.spec_.reward.pit_penalty
        return self.spec_.reward.step_cost

    def _move(self, state: int, action: int) -> int:
        """Where action `action` lands from `state`, ignoring slip.

        Walking into a wall leaves the agent where it was rather than raising
        or wrapping around. Wrapping would make the grid a torus — a different
        problem with a different optimal policy — and is a surprisingly common
        accident when the coordinate maths is written with `%`.
        """
        row, col = self.to_rowcol(state)
        d_row, d_col = _DELTAS[action]
        row = min(max(row + d_row, 0), self.rows - 1)
        col = min(max(col + d_col, 0), self.cols - 1)
        return self.to_index(row, col)

    def _build_model(self):
        """Tabulate P once. This is the object the planner is allowed to see."""
        slip = self.spec_.slip
        if not 0.0 <= slip < 1.0:
            raise ValueError(f"slip must be in [0, 1), got {slip}")

        P: dict[int, dict[int, list[tuple[float, int, float, bool]]]] = {}
        for state in range(self.n_states):
            P[state] = {}
            for action in range(self.n_actions):
                if state in self.terminal_states:
                    # An absorbing self-loop with zero reward. This is not
                    # cosmetic: a terminal state left OUT of P makes every
                    # planner crash on a KeyError, and a terminal state given a
                    # non-zero self-loop reward makes the value function
                    # diverge while every unit test still passes.
                    P[state][action] = [(1.0, state, 0.0, True)]
                    continue

                # The intended direction gets 1 - slip; the two perpendicular
                # directions split the rest. The agent is never blown
                # backwards, which is the FrozenLake convention and keeps the
                # problem solvable at slip = 0.2 without a huge discount.
                outcomes = [
                    (1.0 - slip, action),
                    (slip / 2.0, (action - 1) % 4),
                    (slip / 2.0, (action + 1) % 4),
                ]
                # Collapse duplicate landing squares (they happen in corners,
                # where two directions both bump the same wall). Leaving them
                # split is arithmetically harmless but it makes P rows longer
                # than they need to be and hides the structure when you print
                # one to debug it.
                merged: dict[int, float] = {}
                for prob, direction in outcomes:
                    if prob <= 0.0:
                        continue
                    landing = self._move(state, direction)
                    merged[landing] = merged.get(landing, 0.0) + prob
                P[state][action] = [
                    (
                        prob,
                        next_state,
                        self._reward_fn(state, action, next_state),
                        next_state in self.terminal_states,
                    )
                    for next_state, prob in sorted(merged.items())
                ]
        return P

    def _flatten_model(self):
        """Pack P into rectangular arrays so step() is three array lookups.

        Rows are ragged (a corner state has fewer distinct outcomes than a
        middle one), so the arrays are padded to the widest row and the padding
        is made unreachable by setting the cumulative probability there to 1.0.
        """
        width = max(len(self.P[s][a]) for s in self.P for a in self.P[s])
        shape = (self.n_states, self.n_actions, width)
        cum = np.ones(shape, dtype=np.float64)
        nxt = np.zeros(shape, dtype=np.int64)
        rew = np.zeros(shape, dtype=np.float64)
        done = np.zeros(shape, dtype=bool)
        for s in range(self.n_states):
            for a in range(self.n_actions):
                entries = self.P[s][a]
                probs = np.array([e[0] for e in entries], dtype=np.float64)
                cum[s, a, : len(entries)] = np.cumsum(probs)
                # Pad the tail with 1.0 so searchsorted can never run off the
                # end when floating-point cumulative sums land at 0.999...
                cum[s, a, len(entries) :] = 1.0
                nxt[s, a, : len(entries)] = [e[1] for e in entries]
                nxt[s, a, len(entries) :] = entries[-1][1]
                rew[s, a, : len(entries)] = [e[2] for e in entries]
                rew[s, a, len(entries) :] = entries[-1][2]
                done[s, a, : len(entries)] = [e[3] for e in entries]
                done[s, a, len(entries) :] = entries[-1][3]
        return cum, nxt, rew, done

    # -- the Gymnasium API ---------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Standard reset, plus an `options={"state": s}` escape hatch.

        That escape hatch is what makes EXPLORING STARTS possible. Monte Carlo
        control with exploring starts requires the ability to begin an episode
        in an arbitrary state-action pair, and an environment that only ever
        starts at (0, 0) cannot supply it. Gymnasium reserves `options` for
        exactly this kind of environment-specific request, which is why it is
        not a constructor argument: the start state changes per episode.
        """
        super().reset(seed=seed)
        self._elapsed = 0
        requested = (options or {}).get("state")
        if requested is None:
            self.state = self.start_state
        else:
            requested = int(requested)
            if not 0 <= requested < self.n_states:
                raise ValueError(f"start state {requested} outside [0, {self.n_states})")
            self.state = requested
        return self.state, {"start_state": self.state}

    def step(self, action: int):
        action = int(action)
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action {action} outside [0, {self.n_actions})")

        # np.searchsorted over a precomputed CDF rather than
        # self.np_random.choice(p=probs): choice re-validates and re-normalises
        # the probability vector on every call, which is ~10x the cost of the
        # whole rest of step() and is measurable over a million-step run.
        u = self.np_random.random()
        k = min(int(np.searchsorted(self._cum[self.state, action], u)), self._cum.shape[2] - 1)
        next_state = int(self._next[self.state, action, k])
        reward = float(self._rew[self.state, action, k])
        terminated = bool(self._done[self.state, action, k])

        self.state = next_state
        self._elapsed += 1
        # `truncated` is left to the TimeLimit wrapper applied in make_env().
        # Reporting it here as well would double-count the cap and make the
        # wrapper's max_episode_steps silently unenforceable.
        return next_state, reward, terminated, False, {}

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        out = []
        for row in range(self.rows):
            line = []
            for col in range(self.cols):
                s = self.to_index(row, col)
                if s == self.state:
                    line.append("A")
                elif s == self.goal_state:
                    line.append("G")
                elif s in self.pit_states:
                    line.append("X")
                elif s == self.start_state:
                    line.append("S")
                else:
                    line.append(".")
            out.append(" ".join(line))
        return "\n".join(out)


def make_gridworld(
    spec: GridSpec | None = None,
    reward_fn: Callable[[int, int, int], float] | None = None,
    time_limit: bool = True,
    render_mode: str | None = None,
) -> gym.Env:
    """Build the environment the way every tier in this repository consumes it.

    `time_limit=True` wraps the environment in `gymnasium.wrappers.TimeLimit`,
    which is what turns a policy that never reaches the goal from an infinite
    loop into a truncated episode. Two consequences to hold onto:

      * A truncated return is a BIASED sample of v_pi — the tail is chopped
        off. It is safe here because the optimal policy terminates in ~10 steps
        against a 100-step cap, but `train/monte_carlo.py` logs the truncation
        rate anyway so that the bias is auditable rather than assumed away.
      * The wrapper does not forward `P`. Planners must read `env.unwrapped.P`.
        `time_limit=False` exists for the unit tests, which want the raw object.
    """
    env: gym.Env = GridWorldEnv(spec=spec, reward_fn=reward_fn, render_mode=render_mode)
    if time_limit:
        limit = (spec or GridSpec()).max_episode_steps
        env = gym.wrappers.TimeLimit(env, max_episode_steps=limit)
    return env
