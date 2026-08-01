"""
search/mcts.py — Monte Carlo tree search with UCT, and the PUCT variant the
learned agent uses.

The four phases, and the one thing that goes wrong in each:

  SELECTION      descend the existing tree by the UCB rule until you reach a
                 node with an unexpanded child.
                 *Goes wrong:* the exploration term is computed against the
                 wrong parent visit count, so it never decays and the search
                 never commits.
  EXPANSION      add one child.
                 *Goes wrong:* expanding all children at once, which turns the
                 tree into a breadth-first search with extra bookkeeping.
  SIMULATION     estimate the value of the new node — a random playout here,
                 a network evaluation in the learned agent.
                 *Goes wrong:* the returned value's point of view. This is the
                 bug. See `_backpropagate`.
  BACKPROPAGATION push the result back up the path, flipping sign at every ply.
                 *Goes wrong:* not flipping, which produces an agent that helps
                 its opponent roughly half the time and looks merely "weak".

Why MCTS at all, when alpha-beta is right there and stronger at equal wall clock
on a 7-wide board? Three properties alpha-beta does not have, and they are the
reason this family of algorithms became the basis of AlphaGo and, later, of
search at inference time in reasoning models:

  * It needs no domain evaluation function. A random playout is a valid, if
    noisy, estimate of a position's value in any game whose rules you can
    simulate. Alpha-beta without a good static evaluator is nearly useless.
  * It is anytime. Stop it after 50 simulations or 50,000 and it returns its
    current best answer. Alpha-beta at depth 6 has no answer until depth 6 is
    finished (which is what iterative deepening exists to fix).
  * Its cost is set by a simulation count, not by a branching factor raised to a
    depth. On a game with b = 250 (Go) that is the difference between possible
    and impossible.

The exploration constant C is a constructor argument, not a module constant,
because the whole point of one of this topic's discussion questions is that the
optimal C is problem-dependent — and a constant you cannot sweep is a constant
you will never sweep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from envs.connect_four import COLS, Position, winning_moves
from search.budget import NodeBudget
from search.minimax import SearchStats

# A simulation policy: (position, rng) -> value in [-1, 1] from the point of
# view of the side to move in `position`.
SimulateFn = Callable[[Position, np.random.Generator], float]

# A prior/value function: (position) -> (prior over the 7 columns, value).
# This is what the AlphaZero-inspired agent injects in place of a rollout.
PriorValueFn = Callable[[Position], tuple[np.ndarray, float]]


class Node:
    """One node of the search tree.

    `value_sum` and `visits` are stored from the point of view of the player TO
    MOVE at this node. That convention is arbitrary — the alternative, "always
    from player 1's view", is equally consistent — but it must be picked once
    and obeyed everywhere, and every negation in this file follows from it.

    `__slots__` is not premature optimisation here: a 400-simulation search
    creates a few hundred nodes per move and a benchmark plays thousands of
    moves, and a plain `__dict__` per node is roughly 100 bytes apiece of pure
    bookkeeping. The `games` table has a `peak_kib` column; this is one of the
    reasons the number in it is small.
    """

    __slots__ = (
        "move", "parent", "children", "visits", "value_sum", "prior", "untried",
        "child_priors",
    )

    def __init__(self, move: int | None, parent: "Node | None", untried: list[int],
                 prior: float = 1.0) -> None:
        self.move = move
        self.parent = parent
        self.children: list[Node] = []
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
        self.untried = untried
        # Filled in the first time the network evaluates THIS node's position,
        # and read when this node's children are created. Storing the priors on
        # the parent rather than re-evaluating the network once per child is the
        # difference between one forward pass per node and seven.
        self.child_priors: np.ndarray | None = None

    @property
    def mean_value(self) -> float:
        # An unvisited node has no estimate. Returning 0.0 rather than raising
        # is deliberate — 0 is "drawish", the neutral prior, and it is what the
        # UCB formula wants for a node whose exploration term is about to
        # dominate anyway.
        return self.value_sum / self.visits if self.visits else 0.0

    def is_fully_expanded(self) -> bool:
        return not self.untried


@dataclass
class MCTSResult:
    move: int
    stats: SearchStats
    root_visits: dict[int, int] = field(default_factory=dict)
    root_values: dict[int, float] = field(default_factory=dict)
    value: float = 0.0


# ---------------------------------------------------------------------------
# Simulation policies
# ---------------------------------------------------------------------------


def random_playout(position: Position, rng: np.random.Generator) -> float:
    """Uniform random play to the end. The textbook rollout.

    Returns the result from the point of view of the side to move in the
    position AS PASSED IN, which is the convention `_backpropagate` expects.

    This mutates and then restores `position` with push/pop rather than copying
    it. A playout is up to 42 moves and a 400-simulation search does 400 of
    them per decision; copying the position at the start of each is 400
    allocations per move that a pair of loops avoids entirely.
    """
    to_move = position.player
    pushed = 0
    try:
        while not position.is_terminal():
            moves = position.legal_moves()
            position.push(int(moves[rng.integers(len(moves))]))
            pushed += 1
        return position.result_for(to_move)
    finally:
        # `finally` because an exception mid-playout would otherwise leave the
        # shared position corrupted for every subsequent simulation — a bug that
        # presents as "MCTS gets worse the longer it runs".
        for _ in range(pushed):
            position.pop()


def tactical_playout(position: Position, rng: np.random.Generator) -> float:
    """Random play, except: take a win if one exists, block one if one exists.

    This is the REVISION (see the README's revised-agent section). It is a
    "heavy" playout in the MCTS literature, and the trade it makes is the point:
    each simulation costs roughly 2-3x a uniform playout because `winning_moves`
    runs at every step, so at a fixed wall-clock budget the agent gets fewer
    simulations. It buys a far less noisy estimate, because a uniform playout
    routinely walks past a win and scores a won position as a loss.

    Whether that trade is worth it is an empirical question about THIS game at
    THIS budget, which is precisely why it is measured in `train/benchmark.py`
    rather than argued about.
    """
    to_move = position.player
    pushed = 0
    try:
        while not position.is_terminal():
            me = position.player
            take = winning_moves(position, me)
            if take:
                col = take[0]
            else:
                stop = winning_moves(position, -me)
                if stop:
                    # If the opponent has two distinct winning replies we are
                    # lost whichever we block; blocking the first is as good as
                    # any and keeps the playout cheap.
                    col = stop[0]
                else:
                    moves = position.legal_moves()
                    col = int(moves[rng.integers(len(moves))])
            position.push(col)
            pushed += 1
        return position.result_for(to_move)
    finally:
        for _ in range(pushed):
            position.pop()


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------


class MCTS:
    """UCT, with an optional network prior (PUCT) in place of the rollout.

    One class covers both the plain Monte Carlo agent and the AlphaZero-inspired
    one, for the same reason `search/minimax.py` has one recursion: the two
    differ in what happens at a leaf, and nowhere else. Passing `prior_value`
    switches the selection formula from UCB1 to PUCT and the simulation from a
    playout to a single network evaluation. Every other line is shared.
    """

    def __init__(
        self,
        iterations: int = 400,
        c: float = 1.414,
        *,
        simulate: SimulateFn = random_playout,
        prior_value: PriorValueFn | None = None,
        seed: int | None = None,
        node_budget: int = 200_000,
    ) -> None:
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        self.iterations = int(iterations)
        # sqrt(2) is the value UCB1's regret bound is derived for, when rewards
        # live in [0, 1]. Ours live in [-1, 1], so the theory does not transfer
        # directly and C is a hyperparameter to sweep rather than a constant to
        # trust. The README reports a sweep; do not skip it.
        self.c = float(c)
        self.simulate = simulate
        self.prior_value = prior_value
        self.node_budget = int(node_budget)
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    # -- the four phases ----------------------------------------------------

    def search(self, position: Position) -> MCTSResult:
        """Run `iterations` simulations from `position` and return the best move.

        "Best" is the MOST VISITED child, not the highest mean value. This is
        the standard choice and it is not arbitrary: a child visited three times
        with a lucky mean of 1.0 is a worse bet than one visited two hundred
        times with a mean of 0.6, and the visit count is the statistic the whole
        algorithm was concentrating. Selecting on mean value makes the agent
        maximally sensitive to its own noise.
        """
        import time

        if position.is_terminal():
            raise ValueError("MCTS.search called on a finished game")

        t0 = time.perf_counter()
        budget = NodeBudget(self.node_budget)
        stats = SearchStats()

        root = Node(None, None, self._child_moves(position))
        if self.prior_value is not None:
            # The root is evaluated once, up front, so its children have priors
            # from their first visit. Every other node gets its priors when the
            # simulation phase reaches it.
            root.child_priors, _ = self.prior_value(position)

        for _ in range(self.iterations):
            if budget.exhausted:
                stats.budget_exhausted = True
                break

            node = root
            depth = 0
            pushed = 0

            # --- SELECTION ---------------------------------------------------
            # Descend while the node is fully expanded and not terminal. Note
            # the terminal check: without it the search "expands" a finished
            # game, gets an empty move list, and either crashes or silently
            # treats a won position as a leaf worth whatever the last playout
            # said.
            while node.is_fully_expanded() and node.children and not position.is_terminal():
                # Charge BEFORE taking the step, and stop if the charge fails.
                # Counting first and charging afterwards lets the tree overshoot
                # the budget by one node per level, which is exactly the kind of
                # off-by-a-small-number that makes a "hard" bound soft.
                if not budget.spend(1):
                    stats.budget_exhausted = True
                    break
                node = self._select_child(node)
                position.push(node.move)  # type: ignore[arg-type]
                pushed += 1
                depth += 1
                stats.nodes += 1

            # --- EXPANSION ---------------------------------------------------
            # Same rule: the budget is charged as part of the condition, so an
            # expansion that cannot be paid for simply does not happen and the
            # simulation runs from wherever selection stopped.
            if (not position.is_terminal() and node.untried
                    and not budget.exhausted and budget.spend(1)):
                # Pop a random untried move rather than the first. Taking the
                # first makes the expansion order deterministic and correlated
                # with column index, which biases early estimates towards the
                # left edge of the board — a bias that survives into play at low
                # iteration counts.
                idx = int(self.rng.integers(len(node.untried)))
                move = node.untried.pop(idx)
                # A node whose position the network has not yet seen has no
                # priors, so its children start uniform. That is the honest
                # default: PUCT with a made-up prior is worse than PUCT with a
                # flat one, because a made-up prior is confidently wrong.
                prior = (
                    float(node.child_priors[move])
                    if node.child_priors is not None
                    else 1.0 / COLS
                )
                position.push(move)
                pushed += 1
                depth += 1
                child = Node(move, node, self._child_moves(position), prior=prior)
                node.children.append(child)
                node = child
                stats.nodes += 1

            # --- SIMULATION --------------------------------------------------
            if position.is_terminal():
                # A terminal leaf needs no estimate; it has a value. Using the
                # true result here rather than running a zero-length playout is
                # what lets MCTS actually finish won games.
                value = position.result_for(position.player)
            elif self.prior_value is not None:
                node.child_priors, value = self.prior_value(position)
            else:
                value = self.simulate(position, self.rng)
            stats.leaves += 1

            # --- BACKPROPAGATION ---------------------------------------------
            self._backpropagate(node, value)

            for _ in range(pushed):
                position.pop()
            if depth > stats.max_depth_reached:
                stats.max_depth_reached = depth

        stats.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        stats.budget_exhausted = stats.budget_exhausted or budget.exhausted

        if not root.children:
            # Can only happen with iterations so small that not one expansion
            # completed. Fall back to a legal move rather than raising: the
            # caller is a game in progress.
            moves = position.legal_moves()
            return MCTSResult(move=moves[0], stats=stats, value=0.0)

        visits = {int(ch.move): ch.visits for ch in root.children}
        # Child values are stored from the CHILD's point of view — that is, from
        # the opponent's. Negate to report them from the root player's, which is
        # what the "Play" tab shows a human next to their own board.
        values = {int(ch.move): -ch.mean_value for ch in root.children}
        best = max(root.children, key=lambda ch: (ch.visits, -ch.mean_value))
        stats.root_values = values
        return MCTSResult(
            move=int(best.move),  # type: ignore[arg-type]
            stats=stats,
            root_visits=visits,
            root_values=values,
            value=-best.mean_value,
        )

    # -- the selection rule -------------------------------------------------

    def _select_child(self, node: Node) -> Node:
        """UCB1 (UCT), or PUCT when a network prior is available.

        UCT:   argmax_i  Q_i + C * sqrt( ln N / N_i )
        PUCT:  argmax_i  Q_i + C * P_i * sqrt(N) / (1 + N_i)

        Q_i here is `-child.mean_value`: the child's statistics are from the
        child's mover's point of view, and we are choosing on behalf of the
        parent's mover. Forget this negation and the tree confidently walks into
        the opponent's best lines. It is the single most common MCTS bug and it
        does not announce itself — the agent still plays legal moves.

        The difference between the two formulas is why AlphaZero works: UCT's
        exploration term treats every unvisited child identically, so on a wide
        board it must visit each once before it can prefer any. PUCT weights the
        term by a learned prior, so a network that has seen a million positions
        can tell the search which of 250 moves are worth a first visit at all.
        """
        # ln(0) is -inf; a parent that has never been visited cannot rank its
        # children by anything but the prior, so guard it.
        log_n = math.log(node.visits) if node.visits > 0 else 0.0
        sqrt_n = math.sqrt(node.visits) if node.visits > 0 else 0.0

        best, best_score = None, -float("inf")
        for child in node.children:
            if child.visits == 0:
                # An unvisited child has infinite UCB by construction. Returning
                # it immediately, rather than letting `inf` propagate through
                # the arithmetic, keeps the comparison finite and total.
                return child
            exploit = -child.mean_value
            if self.prior_value is not None:
                explore = self.c * child.prior * sqrt_n / (1 + child.visits)
            else:
                explore = self.c * math.sqrt(log_n / child.visits)
            score = exploit + explore
            if score > best_score:
                best, best_score = child, score
        assert best is not None
        return best

    @staticmethod
    def _backpropagate(node: Node | None, value: float) -> None:
        """Push `value` up the path, flipping sign at every ply.

        `value` arrives from the point of view of the player to move at `node`.
        The parent's mover is the other player, so the same outcome is worth
        `-value` to them. The alternation is the entire content of this
        function, and dropping it produces an agent that is confidently wrong
        on exactly half the plies — which reads as "MCTS is not very good" and
        is in fact "MCTS is being told the wrong sign".
        """
        while node is not None:
            node.visits += 1
            node.value_sum += value
            value = -value
            node = node.parent

    @staticmethod
    def _child_moves(position: Position) -> list[int]:
        return position.legal_moves()
