"""
search/agents.py — the registry. One name, one configured agent, one table.

`POST /act`, `POST /game`, the Streamlit "Play" tab and `train/benchmark.py` all
resolve an agent through `get_agent(name)`. There is no second place where an
agent is constructed, and that is the point: the agent a human plays in the
browser is byte-for-byte the agent the tournament benchmarked, so the win rate
in the `matches` table is a claim about the thing that is deployed.

The alternative — the UI constructing an `MCTS(iterations=400)` because that is
what the slider said, while the benchmark constructed `MCTS(iterations=1000)` —
is not hypothetical. It is the default outcome of not having this file, and it
produces a report whose numbers describe software nobody can play against.

Every agent implements one method:

    choose(position) -> Decision

`Decision` carries the move AND the search telemetry, because in this product
the telemetry is half the deliverable: the node count is what makes the
comparison legible to the non-specialist in the product brief.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from envs.connect_four import Position, evaluate_position, winning_moves
from search.mcts import MCTS, random_playout, tactical_playout
from search.minimax import SearchStats, search_root
from search.ordering import heuristic_ordering, make_beam_ordering, natural_ordering

DEFAULT_NODE_BUDGET = 200_000


@dataclass
class Decision:
    """A move and everything measured while choosing it."""

    move: int
    value: float
    nodes_expanded: int
    wall_clock_ms: float
    search_depth: int
    budget_exhausted: bool = False
    detail: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_stats(cls, move: int, value: float, stats: SearchStats,
                   **detail: Any) -> "Decision":
        return cls(
            move=int(move),
            value=float(value),
            nodes_expanded=stats.nodes,
            wall_clock_ms=stats.elapsed_ms,
            search_depth=stats.max_depth_reached,
            budget_exhausted=stats.budget_exhausted,
            detail=detail,
        )


class Agent:
    """Base class. `name` is what the API and the `games` table record."""

    name = "agent"
    # Human-readable, shown in the UI's agent picker and in the model card. Not
    # decoration: the "Tournament" tab is unreadable without it.
    description = ""

    def choose(self, position: Position) -> Decision:  # pragma: no cover - abstract
        raise NotImplementedError

    def config(self) -> dict[str, Any]:
        """The hyperparameters that determine this agent's decisions.

        Hashed into the `policy_sha256` field of `/act`'s response for agents
        with no artifact. A search agent's identity is its configuration —
        depth, ordering, exploration constant — in exactly the way a neural
        policy's identity is its weights, and the audit log needs SOMETHING that
        distinguishes a depth-4 answer from a depth-6 one six weeks later.
        """
        return {"name": self.name}

    def identity(self) -> str:
        import hashlib
        import json

        blob = json.dumps(self.config(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# The reference opponent
# ---------------------------------------------------------------------------


class RandomAgent(Agent):
    """Uniform over legal columns. The fixed reference the win rates are against.

    Every "decision quality" number in this product is a win rate against THIS
    agent, and it is deliberately the weakest possible opponent that is still
    well defined. Two reasons:

      * It never improves, so a win rate measured in week 5 is comparable with
        one measured in week 12. A win rate against "my previous agent" is a
        moving target and cannot be plotted against anything.
      * It is a floor, not a ceiling. An agent that cannot beat random 100% of
        the time in Connect Four is broken, so this reference catches defects.
        It does NOT separate a good agent from a great one — that is what the
        round-robin between the agents is for, and why the harness reports both.

    Seeded, because "the random agent" with an unseeded RNG makes the whole
    tournament irreproducible.
    """

    name = "random"
    description = "Uniform random legal move. The fixed reference opponent."

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def choose(self, position: Position) -> Decision:
        import time

        t0 = time.perf_counter()
        moves = position.legal_moves()
        move = int(moves[self.rng.integers(len(moves))])
        return Decision(
            move=move,
            value=0.0,
            # One node: itself. Recording 0 would make the mean-nodes column in
            # `matches` divide by a count that includes rows with no search, and
            # "the random agent expands zero nodes" is a claim about arithmetic
            # rather than about search.
            nodes_expanded=1,
            wall_clock_ms=(time.perf_counter() - t0) * 1000.0,
            search_depth=0,
        )

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "seed": self.seed}


# ---------------------------------------------------------------------------
# The one search scaffold, twice
# ---------------------------------------------------------------------------


class MinimaxAgent(Agent):
    """Depth-limited negamax. Exhaustive or heuristic, depending on `ordering`.

    This ONE class is both baseline agents the syllabus asks for. `exhaustive`
    and `heuristic` in the registry below are two instances of it that differ in
    the `ordering` argument and in nothing else — which is the requirement, and
    also the only way the node-count comparison means anything.
    """

    def __init__(
        self,
        name: str,
        depth: int = 4,
        *,
        ordering: Callable[..., list[int]] = natural_ordering,
        alpha_beta: bool = True,
        evaluate: Callable[[Position, int], float] = evaluate_position,
        node_budget: int = DEFAULT_NODE_BUDGET,
        description: str = "",
    ) -> None:
        self.name = name
        self.depth = int(depth)
        self.ordering = ordering
        self.alpha_beta = bool(alpha_beta)
        self.evaluate = evaluate
        self.node_budget = int(node_budget)
        self.description = description or f"negamax depth {depth}"

    def choose(self, position: Position) -> Decision:
        move, value, stats = search_root(
            position,
            self.depth,
            ordering=self.ordering,
            evaluate=self.evaluate,
            alpha_beta=self.alpha_beta,
            node_budget=self.node_budget,
        )
        # `exact_root_values` is deliberately NOT set here, even though the
        # "Play" tab would like exact per-move numbers. Turning it on for
        # interactive requests would mean the agent a human plays expands 2-4x
        # the nodes of the agent the tournament benchmarked, and this file's
        # entire premise is that those are the same agent. The bounds are
        # surfaced honestly instead: `root_values_are_bounds` travels with the
        # decision and the UI labels the numbers accordingly.
        return Decision.from_stats(
            move, value, stats,
            root_values=stats.root_values,
            root_values_are_bounds=stats.root_values_are_bounds,
        )

    def config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "depth": self.depth,
            "ordering": getattr(self.ordering, "__name__", str(self.ordering)),
            "alpha_beta": self.alpha_beta,
            "node_budget": self.node_budget,
        }


class MCTSAgent(Agent):
    """UCT, or PUCT when a network is supplied. See search/mcts.py."""

    def __init__(
        self,
        name: str,
        iterations: int = 400,
        c: float = 1.414,
        *,
        simulate=random_playout,
        prior_value=None,
        seed: int | None = None,
        node_budget: int = DEFAULT_NODE_BUDGET,
        description: str = "",
    ) -> None:
        self.name = name
        self.iterations = int(iterations)
        self.c = float(c)
        self.simulate = simulate
        self.prior_value = prior_value
        self.seed = seed
        self.node_budget = int(node_budget)
        self.description = description or f"MCTS, {iterations} simulations, C={c}"
        self._mcts = MCTS(
            iterations=self.iterations,
            c=self.c,
            simulate=simulate,
            prior_value=prior_value,
            seed=seed,
            node_budget=self.node_budget,
        )

    def choose(self, position: Position) -> Decision:
        result = self._mcts.search(position)
        return Decision.from_stats(
            result.move, result.value, result.stats,
            root_visits=result.root_visits, root_values=result.root_values,
        )

    def config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "c": self.c,
            "simulate": getattr(self.simulate, "__name__", None),
            "network": self.prior_value is not None,
            "seed": self.seed,
            "node_budget": self.node_budget,
        }


class TacticalWrapper(Agent):
    """Take an immediate win; block an immediate loss; otherwise defer to `inner`.

    A two-line safety net in front of any agent. It exists because of a specific
    and very visible failure: a low-iteration MCTS agent, or a beam-pruned
    search, will occasionally decline a win it can see in one move, and a human
    playing it in the browser reads that — correctly — as "this thing is
    broken", regardless of its tournament record.

    It is also a legitimate part of the REVISION story, and an honest one to
    report: it improves decision quality at essentially zero node cost, and it
    does so by hard-coding domain knowledge rather than by searching better. Say
    which of those you did.
    """

    def __init__(self, inner: Agent, name: str | None = None) -> None:
        self.inner = inner
        self.name = name or f"{inner.name}+tactical"
        self.description = f"{inner.description} (with an immediate win/block check)"

    def choose(self, position: Position) -> Decision:
        import time

        t0 = time.perf_counter()
        me = position.player
        for candidate in (winning_moves(position, me), winning_moves(position, -me)):
            if candidate:
                return Decision(
                    move=int(candidate[0]),
                    value=0.0,
                    nodes_expanded=len(position.legal_moves()),
                    wall_clock_ms=(time.perf_counter() - t0) * 1000.0,
                    search_depth=1,
                    detail={"tactical": True},
                )
        return self.inner.choose(position)

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "tactical": True, "inner": self.inner.config()}


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
#
# Depths and iteration counts here are the SANDBOX defaults: small enough that
# `pytest -q` and `python -m train.benchmark --games 30` finish in the time a
# student will actually wait. The README's "real budget" table gives the numbers
# to use for the submitted result, and `train/benchmark.py` takes them as flags
# so you never have to edit this file to run a bigger experiment.

_NETWORK_PATH = pathlib.Path("policies/alphazero_c4.npz")


def _network_prior_value():
    """Load the exported policy-value network, or None if it has not been trained.

    Returns None rather than raising so that a fresh clone — where
    `python -m train.train` has not been run yet — still serves five of the six
    agents. `GET /agents` reports which are unavailable and why, which is a
    better answer than a 500 on the one endpoint the UI calls on page load.
    """
    if not _NETWORK_PATH.exists():
        return None
    try:
        from search.net import PolicyValueNet

        net = PolicyValueNet.from_npz(_NETWORK_PATH)
    except Exception:
        return None
    return net.evaluate


def build_registry(seed: int | None = 0) -> dict[str, Agent]:
    """Construct every agent. Called once per process; see `get_agent`.

    `seed` threads through every stochastic agent so that a whole tournament is
    reproducible from one number. An agent registry with unseeded RNGs produces
    a benchmark that cannot be re-run, which is the same as a benchmark that
    cannot be checked.
    """
    prior_value = _network_prior_value()

    agents: dict[str, Agent] = {}

    def add(agent: Agent) -> None:
        agents[agent.name] = agent

    add(RandomAgent(seed=seed))

    # --- the two baseline searches: same class, different callback -----------
    add(MinimaxAgent(
        "exhaustive",
        depth=4,
        ordering=natural_ordering,
        alpha_beta=False,
        description=(
            "Full-width negamax to depth 4, no pruning, left-to-right. Visits "
            "every node in the tree — the honest baseline everything else is "
            "measured against."
        ),
    ))
    add(MinimaxAgent(
        "heuristic",
        depth=4,
        ordering=heuristic_ordering,
        alpha_beta=True,
        description=(
            "The SAME recursion to depth 4, with alpha-beta and children ordered "
            "by the domain evaluation function. Identical answers to "
            "'exhaustive', a fraction of the nodes."
        ),
    ))
    # The deep variant exists to make the scalability curve interesting and to
    # give the tournament an agent that is genuinely strong. It is the same
    # class again.
    add(MinimaxAgent(
        "heuristic_d6",
        depth=6,
        ordering=heuristic_ordering,
        alpha_beta=True,
        description="Heuristic-ordered alpha-beta at depth 6. The strong reference.",
    ))
    add(MinimaxAgent(
        "beam3",
        depth=6,
        ordering=make_beam_ordering(3),
        alpha_beta=True,
        description=(
            "Forward-pruned: only the best 3 children at each node, to depth 6. "
            "Cheaper than depth 4 full-width and UNSOUND — it can drop a winning "
            "move. In the tournament to show what that costs."
        ),
    ))

    # --- Monte Carlo ---------------------------------------------------------
    add(MCTSAgent(
        "mcts",
        iterations=300,
        c=1.414,
        simulate=random_playout,
        seed=seed,
        description="UCT with uniform random playouts, 300 simulations, C=sqrt(2).",
    ))
    # The REVISION. Same class, two changes, both justified by benchmark
    # evidence in the README: tactical playouts instead of uniform ones, and a
    # lower C because a less noisy value estimate needs less exploration to
    # trust.
    add(MCTSAgent(
        "mcts_v2",
        iterations=300,
        c=0.9,
        simulate=tactical_playout,
        seed=seed,
        description=(
            "REVISED MCTS: win/block-aware playouts and C=0.9. Fewer simulations "
            "per second, much less noise per simulation."
        ),
    ))

    # --- the learned agent ---------------------------------------------------
    if prior_value is not None:
        add(MCTSAgent(
            "alphazero",
            iterations=200,
            c=1.5,
            prior_value=prior_value,
            seed=seed,
            description=(
                "PUCT guided by a policy-value network trained by self-play in "
                "train/selfplay.py and served from a NumPy archive."
            ),
        ))

    return agents


_REGISTRY: dict[str, Agent] | None = None


def get_registry(seed: int | None = 0, refresh: bool = False) -> dict[str, Agent]:
    global _REGISTRY
    if _REGISTRY is None or refresh:
        _REGISTRY = build_registry(seed=seed)
    return _REGISTRY


def agent_names() -> list[str]:
    return list(get_registry().keys())


def get_agent(name: str) -> Agent:
    registry = get_registry()
    if name not in registry:
        raise KeyError(name)
    return registry[name]


def describe_agents() -> list[dict[str, Any]]:
    """Rows for `GET /agents` and for the UI's agent picker."""
    return [
        {
            "name": a.name,
            "description": a.description,
            "config": a.config(),
            "config_sha256": a.identity(),
        }
        for a in get_registry().values()
    ]
