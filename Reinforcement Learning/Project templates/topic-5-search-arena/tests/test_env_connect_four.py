"""
The environment's contract: encoding, masking, gravity, terminal detection.

Every test here guards a property that something else in the repository assumes
without checking. That is the criterion for what belongs in this file — not
"does the code work" but "which assumption elsewhere would break silently".
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from envs.connect_four import (
    COLS,
    DRAW,
    N_CELLS,
    ROWS,
    STATE_DIM,
    IllegalMoveError,
    Position,
    decode_state,
    encode_state,
    evaluate_position,
    winning_moves,
)


# ---------------------------------------------------------------------------
# The JSON round trip. This is the property the whole service depends on.
# ---------------------------------------------------------------------------


def test_state_survives_a_json_round_trip():
    """encode -> json.dumps -> json.loads -> decode reproduces the position.

    The `json` step is not theatre. Pydantic hands back floats for values that
    were written as integers, so a decoder that assumed `int` would work in a
    unit test and fail against the actual API. Going through `json` here is what
    makes this test cover the thing that actually happens on the wire.
    """
    position = Position()
    for col in [3, 3, 2, 4, 4, 2, 5, 1, 0]:
        position.push(col)

    wire = json.loads(json.dumps(encode_state(position)))
    assert len(wire) == STATE_DIM
    assert all(isinstance(v, float) for v in wire), "the wire format is floats"

    restored = decode_state(wire)
    assert restored.board == position.board
    assert restored.player == position.player
    assert restored.heights == position.heights
    assert restored.winner == position.winner


def test_decode_rejects_a_wrong_length_state():
    with pytest.raises(ValueError, match="43"):
        decode_state([0.0] * 10)


def test_decode_rejects_fractional_values():
    """A caller sending 0.5 has a bug, and rounding it silently hides their bug
    inside our answer."""
    state = [0.0] * N_CELLS + [1.0]
    state[0] = 0.5
    with pytest.raises(ValueError, match="fractional"):
        decode_state(state)


def test_decode_rejects_a_floating_piece():
    """Connect Four has gravity. A board with a piece hovering in mid-air is not
    reachable, and searching from it produces confident nonsense."""
    board = [0] * N_CELLS
    board[1 * COLS + 0] = 1        # row 1, column 0, with row 0 empty
    with pytest.raises(ValueError, match="floating"):
        decode_state(board + [-1.0])


def test_decode_rejects_impossible_piece_counts():
    board = [0] * N_CELLS
    board[0] = board[1] = board[2] = 1     # three yellow, no red
    with pytest.raises(ValueError, match="impossible piece counts"):
        decode_state(board + [-1.0])


def test_decode_rejects_a_player_that_contradicts_the_board():
    """The side to move is implied by the piece counts, and the caller's belief
    about it is exactly what a round trip can corrupt."""
    position = Position()
    position.push(3)                       # yellow moved; it is red's turn
    state = encode_state(position)
    state[-1] = 1.0                        # claim it is yellow's turn again
    with pytest.raises(ValueError, match="contradicts the board"):
        decode_state(state)


# ---------------------------------------------------------------------------
# Legal moves and gravity
# ---------------------------------------------------------------------------


def test_action_mask_matches_legal_moves():
    position = Position()
    for _ in range(ROWS):
        position.push(0)
        if position.is_terminal():
            break
    mask = position.action_mask()
    assert mask.dtype == bool and mask.shape == (COLS,)
    assert set(np.flatnonzero(mask).tolist()) == set(position.legal_moves())


def test_a_full_column_is_illegal():
    position = Position()
    # Both sides play column 0 six times. Because the players alternate, the
    # column ends up Y R Y R Y R — full, with nobody having four. Filling it
    # while alternating with a second column instead would give one player all
    # six cells and end the game on the fourth move, which is a fine way to
    # write a test that never reaches the line it was meant to exercise.
    for _ in range(ROWS):
        position.push(0)
    assert 0 not in position.legal_moves()
    with pytest.raises(IllegalMoveError, match="full"):
        position.push(0)


def test_push_pop_restores_the_position_exactly():
    """The invariant the entire search rests on. If it fails, every node count
    and every value in this product is meaningless."""
    position = Position()
    for col in [3, 3, 2, 4]:
        position.push(col)
    snapshot = (list(position.board), list(position.heights), position.player,
                position.n_pieces, position.winner)
    for col in position.legal_moves():
        position.push(col)
        position.pop()
    assert (position.board, position.heights, position.player,
            position.n_pieces, position.winner) == snapshot


# ---------------------------------------------------------------------------
# Terminal detection: the fast incremental path must agree with the slow scan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "moves,expected_winner,label",
    [
        ([3, 0, 4, 1, 5, 2, 6], 1, "horizontal"),
        ([3, 4, 3, 4, 3, 4, 3], 1, "vertical"),
        ([0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3], 1, "diagonal /"),
        ([6, 5, 5, 4, 4, 3, 4, 3, 3, 0, 3], 1, "diagonal \\"),
    ],
)
def test_every_win_direction_is_detected(moves, expected_winner, label):
    position = Position()
    for col in moves:
        position.push(col)
    assert position.winner == expected_winner, f"{label} win not detected"
    # And the incremental detector must agree with the full 69-line scan.
    assert Position(list(position.board), list(position.heights),
                    position.player).winner == expected_winner


# A real 42-move game that ends in a draw, found by random playout and then
# frozen here. Constructing a drawn board by hand is fiddly — most "obviously
# safe" patterns contain a diagonal four you did not notice — and a REACHABLE
# sequence is worth more than a hand-built board anyway, because it also
# exercises `push` all the way to the last cell.
DRAWN_GAME = [5, 5, 5, 0, 5, 4, 4, 6, 6, 3, 5, 1, 3, 5, 2, 1, 2, 3, 0, 4, 1,
              4, 3, 0, 2, 2, 2, 1, 0, 0, 0, 2, 1, 6, 1, 3, 4, 4, 3, 6, 6, 6]


def test_a_full_board_with_no_four_is_a_draw():
    """The draw branch in `push` only fires on the 42nd move.

    Code that runs once per few thousand games is code nobody notices is broken,
    and a draw misreported as a loss would bias every win rate in the product by
    however often draws happen.
    """
    position = Position()
    for col in DRAWN_GAME:
        position.push(col)
    assert position.n_pieces == N_CELLS
    assert position.winner == DRAW, "a full board with no four must be a draw"
    assert position.legal_moves() == []
    assert position.result_for(1) == 0.0 and position.result_for(-1) == 0.0
    # And the incremental detector agrees with the full scan on the final board.
    assert Position(list(position.board), [ROWS] * COLS, player=1).winner == DRAW


def test_terminal_positions_have_no_legal_moves():
    position = Position()
    for col in [3, 0, 4, 1, 5, 2, 6]:
        position.push(col)
    assert position.is_terminal()
    assert position.legal_moves() == []
    with pytest.raises(IllegalMoveError, match="already over"):
        position.push(0)


# ---------------------------------------------------------------------------
# The evaluation function
# ---------------------------------------------------------------------------


def test_evaluation_is_antisymmetric():
    """eval(p, me) == -eval(p, them). A zero-sum game's evaluator must be, and
    a sign error here poisons every negamax value in the product."""
    position = Position()
    for col in [3, 2, 3, 4, 5]:
        position.push(col)
    assert evaluate_position(position, 1) == pytest.approx(-evaluate_position(position, -1))


def test_evaluation_prefers_the_centre_on_an_empty_board():
    empty = Position()
    scores = {}
    for col in range(COLS):
        empty.push(col)
        scores[col] = -evaluate_position(empty, empty.player)
        empty.pop()
    assert max(scores, key=scores.get) == COLS // 2


def test_winning_moves_finds_every_immediate_win():
    position = Position()
    for col in [2, 0, 3, 6]:            # yellow at 2,3; red at 0,6
        position.push(col)
    position.push(4)                    # yellow at 2,3,4 — two open ends
    position.push(0)                    # red replies elsewhere
    assert sorted(winning_moves(position, 1)) == [1, 5]


def test_winning_moves_leaves_the_board_untouched():
    """It writes into `board` and writes back; a leak here corrupts the search
    silently, and the symptom is an agent that gets worse the longer it runs."""
    position = Position()
    for col in [2, 0, 3, 6, 4, 0]:
        position.push(col)
    before = list(position.board)
    winning_moves(position, 1)
    winning_moves(position, -1)
    assert position.board == before
