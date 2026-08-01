"""
The required test: exhaustive search finds a KNOWN FORCED WIN on a CONSTRUCTED
position.

"Constructed" is the operative word. A test that plays a game and checks the
agent won proves nothing — a random agent wins sometimes. A test that hands the
search a position whose value is known by hand, and demands the one move that
realises it, is a test that fails when the search is wrong.

    row 0:   R . Y Y . . R          yellow to move
             0 1 2 3 4 5 6

Yellow plays column 4, making Y Y Y at columns 2, 3, 4 with BOTH ends open and
both landing squares on the bottom row. Red can block column 1 or column 5, not
both, so yellow wins on the next move whatever red does. That is a forced win in
three plies, and column 4 is the only move that produces one — every other
yellow move makes at most a single threat, which red simply blocks.

The value the search must return is `WIN_SCORE - 3`: a win, discounted by the
three plies it takes. That number is not incidental. It is what makes the agent
prefer mate in three to mate in five, and without it an agent holding a won
position shuffles pieces indefinitely, which a human opponent reads — correctly
— as an agent that has stopped trying.
"""

from __future__ import annotations

import pytest

from envs.connect_four import WIN_SCORE, Position
from search.agents import get_agent
from search.minimax import search_root
from search.ordering import heuristic_ordering, make_beam_ordering, natural_ordering

FORCED_WIN_SETUP = [2, 0, 3, 6]      # yellow -> 2, 3 ; red -> 0, 6
WINNING_COLUMN = 4
PLIES_TO_WIN = 3


def forced_win_position() -> Position:
    position = Position()
    for col in FORCED_WIN_SETUP:
        position.push(col)
    assert position.player == 1, "the constructed position must have yellow to move"
    assert not position.is_terminal()
    return position


def test_the_constructed_position_has_no_win_in_one():
    """Guards the test itself.

    If a later edit made column 4 an immediate win, this test would still pass
    while checking something trivial. Asserting that NO move wins immediately is
    what keeps it a test about lookahead.
    """
    from envs.connect_four import winning_moves

    position = forced_win_position()
    assert winning_moves(position, 1) == []
    assert winning_moves(position, -1) == []


def test_exhaustive_search_finds_the_forced_win():
    """The required assertion: build the position, assert the move."""
    position = forced_win_position()
    move, value, stats = search_root(
        position, PLIES_TO_WIN,
        ordering=natural_ordering, alpha_beta=False, node_budget=10**7,
    )
    assert move == WINNING_COLUMN, (
        f"exhaustive search played column {move}; the forced win is column "
        f"{WINNING_COLUMN}. Root values: {stats.root_values}"
    )
    assert value == pytest.approx(WIN_SCORE - PLIES_TO_WIN), (
        "the value must be a win discounted by the three plies it takes — see "
        "search/minimax.py's depth shading"
    )
    # And it must be the ONLY winning move at this depth, or the assertion above
    # is weaker than it looks.
    winners = [c for c, v in stats.root_values.items() if v > WIN_SCORE / 2]
    assert winners == [WINNING_COLUMN]


def test_alpha_beta_and_ordering_find_the_same_forced_win_for_fewer_nodes():
    """Same recursion, different ordering callback. Same answer, fewer nodes.

    This is the syllabus's shared-scaffold requirement stated as a test: if the
    two ever disagree, the scaffold has stopped being shared.
    """
    baseline_move, baseline_value, baseline = search_root(
        forced_win_position(), PLIES_TO_WIN,
        ordering=natural_ordering, alpha_beta=False, node_budget=10**7,
    )
    move, value, stats = search_root(
        forced_win_position(), PLIES_TO_WIN,
        ordering=heuristic_ordering, alpha_beta=True, node_budget=10**7,
    )
    assert (move, value) == (baseline_move, baseline_value)
    assert stats.nodes < baseline.nodes
    assert stats.cutoffs > 0, "alpha-beta that never cuts off is not alpha-beta"


def test_depth_shading_scores_a_faster_win_higher():
    """A win in one must score strictly higher than a win in three.

    That ordering is what makes an agent finish a won game. Without it the
    search is indifferent between mating now and mating in four, and against a
    human it shuffles pieces while holding a winning position — which reads as
    an agent that has given up, and occasionally shuffles into a draw.

    Both searches run to depth 5, so the difference cannot come from the search
    depth: it comes from the ply at which the terminal node was found, which is
    exactly what the shading term encodes.
    """
    win_in_three = forced_win_position()
    _, value_3, _ = search_root(win_in_three, 5, ordering=natural_ordering,
                                alpha_beta=False, node_budget=10**7)

    win_in_one = forced_win_position()
    win_in_one.push(WINNING_COLUMN)     # yellow makes the double threat
    win_in_one.push(0)                  # red blocks neither end
    _, value_1, _ = search_root(win_in_one, 5, ordering=natural_ordering,
                                alpha_beta=False, node_budget=10**7)

    assert value_1 == pytest.approx(WIN_SCORE - 1)
    assert value_3 == pytest.approx(WIN_SCORE - PLIES_TO_WIN)
    assert value_1 > value_3


def test_alpha_beta_root_values_are_bounds_unless_asked_otherwise():
    """The subtlety the "Play" tab depends on getting right.

    With pruning on, a root child that fails low returns an upper bound rather
    than a value — it stopped as soon as it knew the move could not win. The
    BEST move's value is always exact; the others are not, unless
    `exact_root_values=True` pays for them. A UI that shows a human seven
    numbers had better know which kind it has.
    """
    truth = search_root(forced_win_position(), 5, ordering=natural_ordering,
                        alpha_beta=False, node_budget=10**7)[2]
    pruned = search_root(forced_win_position(), 5, ordering=heuristic_ordering,
                         alpha_beta=True, node_budget=10**7)[2]
    exact = search_root(forced_win_position(), 5, ordering=heuristic_ordering,
                        alpha_beta=True, exact_root_values=True, node_budget=10**7)[2]

    assert truth.root_values_are_bounds is False
    assert pruned.root_values_are_bounds is True
    assert exact.root_values_are_bounds is False

    # Exact mode reproduces the unpruned values move for move...
    assert exact.root_values == pytest.approx(truth.root_values)
    # ...and costs more nodes than the pruned mode, which is what it bought.
    assert exact.nodes > pruned.nodes
    # ...while the pruned mode disagrees on at least one non-best move, which is
    # the whole reason this distinction is documented rather than assumed away.
    assert pruned.root_values != pytest.approx(truth.root_values)


def test_forward_pruning_is_unsound_and_this_is_the_proof():
    """A beam over a bad ranking MISSES the forced win the exhaustive search finds.

    This is not a bug being documented as a feature. It is the property that
    distinguishes reordering from forward pruning, and the README's claim that a
    beam search is "a guess with a smaller bill" needs a demonstration rather
    than an assurance. A beam of width 2 over the LEFT-TO-RIGHT ordering only
    ever considers columns 0 and 1, so column 4 is unreachable no matter how
    deep it searches.
    """
    move, value, stats = search_root(
        forced_win_position(), PLIES_TO_WIN,
        ordering=make_beam_ordering(2, inner=natural_ordering),
        alpha_beta=True, node_budget=10**7,
    )
    assert move != WINNING_COLUMN
    assert value < WIN_SCORE / 2, "the beam must not have found the win"
    assert stats.nodes < 40, "and it must be much cheaper — that is the whole trade"


def test_the_registered_agents_find_it_too():
    """The agents a human can actually play must pass the same check.

    `search_root` passing while the registered agent fails would mean the
    registry configured something the tests never exercise — which is precisely
    how a demo ends up weaker than the code it was built from.
    """
    for name in ("exhaustive", "heuristic", "heuristic_d6"):
        decision = get_agent(name).choose(forced_win_position())
        assert decision.move == WINNING_COLUMN, f"agent '{name}' missed the forced win"
