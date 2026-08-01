"""
alpha-beta must return the IDENTICAL value to plain minimax, and heuristic
ordering must reduce the node count.

Budget note: this file checks 120 positions, which is enough to catch a broken
cutoff and fast enough to run on every commit. The claim in the README rests on
1,000 positions from `python -m train.verify`, which takes about a minute. Both
numbers are stated in the README; a test suite that takes a minute is a test
suite people stop running.

Why `==` and not `pytest.approx`: alpha-beta and full-width minimax perform the
same arithmetic over the same values in a different ORDER, and floating-point
addition is not associative — but nothing here is added. Values propagate by
negation and maximum only, so the results are bit-identical, and a tolerance
would hide the one-in-a-hundred wrong value that a bad cutoff produces.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.connect_four import Position
from search.minimax import plain_minimax_value, search_root
from search.ordering import heuristic_ordering, natural_ordering

POSITIONS = 120
DEPTH = 4


def random_positions(n: int, seed: int = 0, max_plies: int = 14) -> list[Position]:
    """Reachable, non-terminal positions from random play.

    Random PLAY, not random board fills: a randomly filled board is usually
    unreachable, and a value computed from an unreachable position is a value
    about a game nobody is playing.
    """
    rng = np.random.default_rng(seed)
    out: list[Position] = []
    while len(out) < n:
        position = Position()
        for _ in range(int(rng.integers(0, max_plies + 1))):
            moves = position.legal_moves()
            if not moves:
                break
            position.push(int(moves[rng.integers(len(moves))]))
            if position.is_terminal():
                break
        if not position.is_terminal():
            out.append(position)
    return out


@pytest.fixture(scope="module")
def sample() -> list[Position]:
    return random_positions(POSITIONS, seed=20240501)


def test_alpha_beta_equals_plain_minimax(sample):
    mismatches = []
    for position in sample:
        reference = plain_minimax_value(position.copy(), DEPTH)
        _, value, _ = search_root(
            position.copy(), DEPTH,
            ordering=natural_ordering, alpha_beta=True, node_budget=10**9,
        )
        if value != reference:
            mismatches.append((list(position.history), reference, value))
    assert not mismatches, (
        f"{len(mismatches)}/{len(sample)} positions disagree. First: {mismatches[0]}. "
        "A wrong cutoff does not crash — it returns the wrong value only where it "
        "fires, which is why this check exists at all."
    )


def test_heuristic_ordering_is_value_preserving(sample):
    """Reordering children cannot change the value. Dropping them can.

    This is what separates `heuristic_ordering` (a permutation) from
    `make_beam_ordering` (a subset), and it is the reason the exhaustive and
    heuristic agents can be compared at all.
    """
    for position in sample:
        reference = plain_minimax_value(position.copy(), DEPTH)
        _, value, _ = search_root(
            position.copy(), DEPTH,
            ordering=heuristic_ordering, alpha_beta=True, node_budget=10**9,
        )
        assert value == reference


def test_ordering_reduces_nodes_under_alpha_beta(sample):
    """The point of the ordering callback, as a number.

    Averaged over the sample rather than checked per position: on a position
    with one legal move the two orderings are trivially equal, and a per-position
    strict inequality would be a test that fails for a correct implementation.
    """
    natural_total = ordered_total = 0
    for position in sample:
        _, _, s_nat = search_root(position.copy(), DEPTH, ordering=natural_ordering,
                                  alpha_beta=True, node_budget=10**9)
        _, _, s_ord = search_root(position.copy(), DEPTH, ordering=heuristic_ordering,
                                  alpha_beta=True, node_budget=10**9)
        natural_total += s_nat.nodes
        ordered_total += s_ord.nodes
    assert ordered_total < natural_total, (
        f"heuristic ordering expanded {ordered_total} nodes against "
        f"{natural_total} for left-to-right. An ordering that does not reduce "
        "the node count is usually sorted the wrong way round."
    )


def test_ordering_alone_changes_nothing_without_pruning(sample):
    """The result students find surprising, pinned as a test.

    A full-width search visits every node whatever order it visits them in, so
    "heuristic search" with alpha-beta OFF costs exactly what exhaustive search
    costs. Ordering is not a search strategy; it is a multiplier on alpha-beta.
    If this test ever fails, the ordering callback has started dropping moves and
    the "exhaustive" agent is no longer exhaustive.
    """
    for position in sample[:20]:
        _, _, s_nat = search_root(position.copy(), DEPTH, ordering=natural_ordering,
                                  alpha_beta=False, node_budget=10**9)
        _, _, s_ord = search_root(position.copy(), DEPTH, ordering=heuristic_ordering,
                                  alpha_beta=False, node_budget=10**9)
        assert s_nat.nodes == s_ord.nodes
        assert s_nat.cutoffs == 0 and s_ord.cutoffs == 0


def test_alpha_beta_takes_cutoffs(sample):
    """Guards against alpha-beta being accidentally disabled.

    Without this, every equivalence test above would still pass with the cutoff
    branch deleted — they would just be comparing full-width minimax against
    itself, at full cost, forever.
    """
    total = 0
    for position in sample[:20]:
        _, _, stats = search_root(position.copy(), DEPTH, ordering=natural_ordering,
                                  alpha_beta=True, node_budget=10**9)
        total += stats.cutoffs
    assert total > 0
