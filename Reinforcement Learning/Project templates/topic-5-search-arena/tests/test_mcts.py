"""
MCTS must beat random by a wide margin at a high iteration budget.

This is a STATISTICAL test, which makes it a different kind of object from the
rest of the suite, and it is worth being explicit about how it is kept from
being flaky:

  * every RNG is seeded, so the run is deterministic given the code;
  * the threshold (85%) is far below the true win rate (~99% at this budget),
    so a change that moves the number by a few points does not fail the build;
  * the game count is small enough to run in seconds and large enough that a
    genuinely broken agent — one with the sign error in backpropagation, say,
    which wins about 50% — is nowhere near the threshold.

The README's 90%-over-100-games claim comes from `python -m train.verify
--mcts-games 100 --mcts-simulations 800`, which is the same check at a budget
worth quoting. This file is the fast guard; that is the evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.connect_four import Position
from search.agents import MCTSAgent, RandomAgent, get_agent
from search.mcts import MCTS, random_playout, tactical_playout

GAMES = 24
SIMULATIONS = 250
THRESHOLD = 0.85


def play_series(agent, opponent, games: int, seed: int = 0) -> dict:
    """Alternating colours, from `agent`'s point of view.

    Alternating is not optional: Connect Four is a first-player win under
    perfect play, so a series in which the agent always moves first measures the
    opening as much as it measures the agent.
    """
    wins = draws = losses = 0
    for g in range(games):
        agent_is_yellow = (g % 2 == 0)
        if hasattr(agent, "_mcts"):
            agent._mcts.rng = np.random.default_rng(seed + g)
        opponent.rng = np.random.default_rng(seed + 10_000 + g)
        position = Position()
        while not position.is_terminal():
            mine = (position.player == 1) == agent_is_yellow
            position.push((agent if mine else opponent).choose(position).move)
        if position.winner == 0:
            draws += 1
        elif position.winner == (1 if agent_is_yellow else -1):
            wins += 1
        else:
            losses += 1
    return {"wins": wins, "draws": draws, "losses": losses,
            "win_rate": (wins + 0.5 * draws) / games}


def test_mcts_beats_random_by_a_wide_margin():
    agent = MCTSAgent("mcts", iterations=SIMULATIONS, c=1.414, seed=7)
    result = play_series(agent, RandomAgent(seed=11), GAMES, seed=7)
    assert result["win_rate"] >= THRESHOLD, (
        f"MCTS at {SIMULATIONS} simulations scored {result['win_rate']:.3f} "
        f"against random ({result}). A rate near 0.5 is the signature of the "
        "missing negation in _backpropagate; a rate near 0 means the sign is "
        "inverted rather than dropped."
    )


def test_more_simulations_do_not_make_it_worse():
    """A weak monotonicity check that catches a whole class of bug.

    Any defect that makes extra search actively harmful — a leaked position from
    a playout, a sign flip in selection, a node reused across searches — shows up
    here and is invisible in a single-budget test.
    """
    low = play_series(MCTSAgent("mcts", iterations=25, c=1.414, seed=3),
                      RandomAgent(seed=5), 16, seed=3)
    high = play_series(MCTSAgent("mcts", iterations=400, c=1.414, seed=3),
                       RandomAgent(seed=5), 16, seed=3)
    assert high["win_rate"] >= low["win_rate"]


def test_playouts_leave_the_position_untouched():
    """The invariant that makes push/pop playouts safe.

    A playout that leaks one move corrupts every later simulation in the same
    search, and the symptom is an agent that gets WORSE the longer it thinks —
    a bug people spend days on because the obvious hypothesis is the opposite.
    """
    position = Position()
    for col in [3, 3, 2, 4]:
        position.push(col)
    snapshot = (list(position.board), list(position.heights), position.player,
                position.n_pieces)
    rng = np.random.default_rng(0)
    for _ in range(50):
        random_playout(position, rng)
        tactical_playout(position, rng)
    assert (position.board, position.heights, position.player,
            position.n_pieces) == snapshot


def test_mcts_takes_an_immediate_win():
    """The cheapest sanity check a human opponent will apply within two moves."""
    position = Position()
    for col in [2, 0, 3, 6, 4, 0]:      # yellow threatens at columns 1 and 5
        position.push(col)
    result = MCTS(iterations=200, seed=0).search(position)
    assert result.move in (1, 5)


def test_mcts_blocks_an_immediate_loss():
    position = Position()
    for col in [2, 3, 0, 4, 6, 5]:      # red now has 3,4,5 with 2 and 6 open
        position.push(col)
    assert position.player == 1
    result = MCTS(iterations=400, seed=0).search(position)
    assert result.move in (1, 6), f"MCTS played {result.move}, ignoring a mate in one"


def test_exploration_constant_changes_the_search():
    """C must actually do something, or a sweep over it is a sweep over noise."""
    position = Position()
    position.push(3)
    greedy = MCTS(iterations=200, c=0.0, seed=0).search(position)
    explorer = MCTS(iterations=200, c=8.0, seed=0).search(position)
    spread = lambda r: max(r.root_visits.values()) - min(r.root_visits.values())  # noqa: E731
    assert spread(greedy) > spread(explorer), (
        "a large C should spread visits more evenly across the root's children; "
        "if it does not, the exploration term is not reaching the selection rule"
    )


def test_the_revised_agent_beats_the_original_head_to_head():
    """The revision claim, as a test rather than as a paragraph.

    `mcts_v2` differs from `mcts` in its playout policy and its exploration
    constant. The README argues that trade is worth it; this asserts the
    direction of the effect at a small budget so that a change which silently
    reverses it fails the build. The MAGNITUDE belongs to the benchmark, not
    here — 20 games has a standard error of about 11 points.
    """
    original, revised = get_agent("mcts"), get_agent("mcts_v2")
    wins = 0
    games = 20
    for g in range(games):
        revised_is_yellow = (g % 2 == 0)
        revised._mcts.rng = np.random.default_rng(g)
        original._mcts.rng = np.random.default_rng(1000 + g)
        position = Position()
        while not position.is_terminal():
            mine = (position.player == 1) == revised_is_yellow
            position.push((revised if mine else original).choose(position).move)
        if position.winner == 0:
            wins += 0.5
        elif position.winner == (1 if revised_is_yellow else -1):
            wins += 1
    assert wins / games > 0.5, (
        f"the revised agent scored {wins}/{games} against the original. If this "
        "fails, the revision is not an improvement and the README must say so "
        "rather than the test being relaxed."
    )


@pytest.mark.parametrize("iterations", [1, 2, 5])
def test_a_tiny_budget_still_returns_a_legal_move(iterations):
    """The degenerate case. A game in a browser cannot be allowed to 500 because
    the simulation slider was dragged to the left."""
    position = Position()
    for col in [3, 3, 2]:
        position.push(col)
    result = MCTS(iterations=iterations, seed=0).search(position)
    assert result.move in position.legal_moves()
