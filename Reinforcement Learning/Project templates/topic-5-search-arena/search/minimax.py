"""
search/minimax.py — ONE recursion, four search strategies.

This is the file the topic is built around, so read the argument before the
code.

The product ships an "exhaustive" search and a "heuristic" search. The naive way
to build that is two functions. It is also the wrong way, and the syllabus makes
sharing a scaffold a requirement rather than a suggestion, for three reasons
that are worth internalising well beyond this assignment:

  1. **Two implementations produce two sets of bugs.** The single most common
     failure in game-search coursework is a sign error, and a sign error in one
     of two copies produces a heuristic agent that is worse than exhaustive for
     a reason that looks like "heuristics are worse" rather than like a defect.
     With one recursion, a sign error breaks both agents identically and is
     found in an afternoon.

  2. **The comparison stops being a comparison.** If the two differ in the
     recursion as well as the ordering, a node-count difference cannot be
     attributed to the ordering. The experiment has a confound built into its
     source code.

  3. **It names the actual idea.** Exhaustive search and heuristic search are
     not two algorithms. They are one algorithm — depth-limited negamax — under
     two policies for which child to look at next. Writing them as one function
     with an injected callback is the code saying so.

So: `negamax()` below is the whole search. What varies is

    ordering    : which children, in what order      (search/ordering.py)
    alpha_beta  : whether cutoffs are taken          (a bool)
    evaluate    : what a depth-limited leaf is worth (a callback)

and nothing else.

NEGAMAX, NOT MINIMAX. Connect Four is zero-sum, so `min(a, b) == -max(-a, -b)`
and the two-function minimax/maximin pair collapses into one function that
always maximises and negates on the way back. That halves the code and removes
the other classic bug — a `min` node that got a `max` body during a copy-paste.
The cost is that every value is "from the side to move's point of view", and you
must hold that in your head. Every return in this file obeys it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from envs.connect_four import WIN_SCORE, Position, evaluate_position
from search.budget import NodeBudget
from search.ordering import OrderingFn, natural_ordering

# A leaf evaluator: (position, player) -> score from `player`'s point of view.
EvaluatorFn = Callable[[Position, int], float]

# Terminal scores are shaded by depth so that the search PREFERS A FASTER WIN and
# a SLOWER LOSS. Without this the agent is indifferent between mating now and
# mating in four, which against a human looks like an agent that has stopped
# trying — it shuffles pieces while holding a won position, and eventually
# shuffles into a draw. One term, large behavioural difference.
_DEPTH_SHADE = 1.0


@dataclass
class SearchStats:
    """Everything the benchmark, the API and the UI need to know about one search.

    These fields are the schema of the `games` table's search columns, and they
    are collected here rather than being recomputed later because a node count
    reconstructed from a log is a node count nobody can check.
    """

    nodes: int = 0                    # every call to negamax, including the root
    leaves: int = 0                   # nodes scored by `evaluate` rather than recursed
    cutoffs: int = 0                  # beta cutoffs — zero when alpha_beta is off
    max_depth_reached: int = 0
    budget_exhausted: bool = False
    elapsed_ms: float = 0.0
    root_values: dict[int, float] = field(default_factory=dict)
    # Whether the numbers in `root_values` are exact values or upper bounds.
    # See `search_root`'s `exact_root_values` argument — this flag exists so a
    # caller that displays them (the "Play" tab) cannot forget which it has.
    root_values_are_bounds: bool = False

    def as_row(self) -> dict[str, object]:
        return {
            "nodes_expanded": self.nodes,
            "leaves": self.leaves,
            "cutoffs": self.cutoffs,
            "search_depth": self.max_depth_reached,
            "budget_exhausted": self.budget_exhausted,
            "wall_clock_ms": round(self.elapsed_ms, 3),
        }


def negamax(
    position: Position,
    depth: int,
    alpha: float,
    beta: float,
    *,
    ordering: OrderingFn,
    evaluate: EvaluatorFn,
    budget: NodeBudget,
    stats: SearchStats,
    alpha_beta: bool,
    ply: int = 0,
) -> float:
    """Value of `position` for the side to move, searching `depth` more plies.

    `alpha` and `beta` are always passed, even when `alpha_beta` is False. They
    are simply never used to cut off in that case. The alternative — a second
    recursion without the window — is the duplication this whole file exists to
    avoid, and the cost of carrying two unused floats is nothing.
    """
    stats.nodes += 1
    if ply > stats.max_depth_reached:
        stats.max_depth_reached = ply

    # Terminal first, ALWAYS before the depth check. A won position at the
    # horizon is worth a win, not whatever the heuristic thinks of the shape;
    # reversing these two branches makes an agent that walks into mate-in-one
    # because the resulting board "looks fine".
    if position.is_terminal():
        stats.leaves += 1
        result = position.result_for(position.player)
        if result == 0.0:
            return 0.0
        # Shade by ply so a win now beats a win later. Note the sign: a LOSS is
        # made less bad by being further away, which is the same term.
        return result * (WIN_SCORE - _DEPTH_SHADE * ply)

    if depth <= 0:
        stats.leaves += 1
        return evaluate(position, position.player)

    moves = position.legal_moves()
    ordered = ordering(position, moves, depth)

    best = -float("inf")
    for col in ordered:
        # Charge the budget BEFORE descending, so the bound holds on the way
        # down rather than being discovered on the way back up. When it runs
        # out we stop recursing and score statically: the caller still gets a
        # legal move, flagged as truncated. See search/budget.py.
        if not budget.spend(1):
            stats.budget_exhausted = True
            position.push(col)
            value = -evaluate(position, position.player)
            position.pop()
        else:
            position.push(col)
            value = -negamax(
                position,
                depth - 1,
                -beta,
                -alpha,
                ordering=ordering,
                evaluate=evaluate,
                budget=budget,
                stats=stats,
                alpha_beta=alpha_beta,
                ply=ply + 1,
            )
            position.pop()

        if value > best:
            best = value
        if alpha_beta:
            if best > alpha:
                alpha = best
            if alpha >= beta:
                # A beta cutoff: the opponent already has a reply at least this
                # good elsewhere, so the true value of this subtree cannot
                # affect the choice above us. Note `>=` and not `>` — with `>`
                # the search is still correct but stops taking cutoffs on equal
                # values, which is most of them in a game with many draws.
                stats.cutoffs += 1
                break

    return best


def search_root(
    position: Position,
    depth: int,
    *,
    ordering: OrderingFn = natural_ordering,
    evaluate: EvaluatorFn = evaluate_position,
    alpha_beta: bool = True,
    node_budget: int = 200_000,
    strict_budget: bool = False,
    exact_root_values: bool = False,
) -> tuple[int, float, SearchStats]:
    """Choose a move. Returns `(column, value, stats)`.

    The root is written out rather than reusing `negamax` with a "return the
    best move too" flag, because a recursion that sometimes returns a move and
    sometimes a value is a recursion whose type nobody can hold in their head.
    Twelve duplicated lines here buy a recursion with one return type.

    `value` is from the point of view of the player to move in `position`, and
    it is EXACT whether or not alpha-beta is on.

    **`stats.root_values` is a different matter, and this is the subtlety in
    this file.** With alpha-beta on, each root child is searched with the
    window `(alpha, +inf)` where alpha is the best value found so far. A child
    whose true value is below alpha "fails low" and returns an UPPER BOUND, not
    a value — it stopped as soon as it knew the child could not win, which is
    the entire point of the pruning. So the per-move numbers for the losing
    moves are bounds, and displaying them next to a board as though they were
    values would be showing a human a number the search never computed.

    `exact_root_values=True` searches every root child with a full window,
    which makes all seven numbers exact at the cost of the alpha propagation
    between root children — typically 2-4x more nodes at depth 6. That is the
    right trade for the "Play" tab, which searches one position per human move
    and cares about explaining itself, and the wrong trade for a tournament,
    which plays thousands of positions and only needs the move. Hence a flag
    rather than a decision.

    `stats.root_values_are_bounds` records which you got.

    On an exhausted budget this still returns a legal move and
    `stats.budget_exhausted is True`. Callers that need the strict behaviour —
    the scalability probe, which must distinguish "did not fit" from "answered
    badly" — pass `strict_budget=True` and catch `NodeBudgetExceeded`.
    """
    import time

    if position.is_terminal():
        raise ValueError("search_root called on a finished game")
    if depth < 1:
        raise ValueError("depth must be at least 1 to choose a move")

    budget = NodeBudget(node_budget, strict=strict_budget)
    stats = SearchStats()
    t0 = time.perf_counter()

    moves = position.legal_moves()
    ordered = ordering(position, moves, depth)
    stats.nodes += 1                                  # the root is a node we visited

    alpha, beta = -float("inf"), float("inf")
    best_move, best_value = ordered[0], -float("inf")

    for col in ordered:
        # Same rule as inside the recursion: charge before descending, and if
        # the budget is gone, score this child statically instead. The root
        # loop must obey it too — a root that keeps recursing after exhaustion
        # can overshoot the budget by one node per remaining column, which is
        # exactly the off-by-b that makes a "hard" bound soft.
        if not budget.spend(1):
            stats.budget_exhausted = True
            position.push(col)
            value = -evaluate(position, position.player)
            position.pop()
        else:
            # `child_alpha` is where the exactness decision is spent. Passing
            # `alpha` propagates the best-so-far and prunes hard; passing -inf
            # gives an exact value for every child.
            child_alpha = -float("inf") if exact_root_values else alpha
            position.push(col)
            value = -negamax(
                position,
                depth - 1,
                -beta,
                -child_alpha,
                ordering=ordering,
                evaluate=evaluate,
                budget=budget,
                stats=stats,
                alpha_beta=alpha_beta,
                ply=1,
            )
            position.pop()
        stats.root_values[col] = value

        if value > best_value:
            best_value, best_move = value, col
        if alpha_beta and best_value > alpha:
            alpha = best_value
        # NOTE: no cutoff at the root. `alpha >= beta` cannot fire here because
        # beta is +inf; the root loop must visit every legal move, because the
        # root's children are the moves we are choosing between.

    stats.root_values_are_bounds = alpha_beta and not exact_root_values
    stats.budget_exhausted = stats.budget_exhausted or budget.exhausted
    stats.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if stats.max_depth_reached < 1:
        stats.max_depth_reached = 1
    return best_move, best_value, stats


def plain_minimax_value(
    position: Position,
    depth: int,
    *,
    evaluate: EvaluatorFn = evaluate_position,
    ply: int = 0,
) -> float:
    """The reference implementation: full width, no pruning, no ordering.

    Exists for exactly one purpose — `tests/test_alpha_beta.py` asserts that
    alpha-beta returns the identical value to this over a thousand random
    positions. That is the only check that catches an incorrectly implemented
    cutoff, because a wrong cutoff does not crash and does not obviously
    misplay; it just quietly returns the wrong value in a small fraction of
    positions, in exactly the positions where the cutoff fired.

    Written out separately from `negamax` on purpose: a "reference" that shares
    the code under test cannot detect a bug in the code it shares. But it must
    share the LEAF CONVENTION — the same evaluator and the same depth-shaded
    terminal score — or the test compares two different functions and fails for
    a reason that has nothing to do with pruning. That is why `ply` is threaded
    through here even though nothing else needs it.
    """
    if position.is_terminal():
        result = position.result_for(position.player)
        if result == 0.0:
            return 0.0
        return result * (WIN_SCORE - _DEPTH_SHADE * ply)
    if depth <= 0:
        return evaluate(position, position.player)
    best = -float("inf")
    for col in position.legal_moves():
        position.push(col)
        value = -plain_minimax_value(position, depth - 1, evaluate=evaluate, ply=ply + 1)
        position.pop()
        if value > best:
            best = value
    return best
