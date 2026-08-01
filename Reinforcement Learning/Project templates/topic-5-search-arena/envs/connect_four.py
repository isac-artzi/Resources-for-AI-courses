"""
envs/connect_four.py — the rules of the game. Pure Python and NumPy, nothing else.

This module imports NO gymnasium and NO framework, and that is enforced in CI
(see .github/workflows/ci.yml, "The search path imports NumPy and nothing
else"). The Gymnasium wrapper lives next door in `envs/gym_env.py` and is
imported lazily by `make_env()`, so the service tier — which needs the rules and
the search but never constructs an environment — loads neither.

`Position` is the SEARCH data structure. It is a mutable board with `push(col)`
/ `pop()`, because a search that copies the board at every node spends most of
its time in the allocator. At depth 6 a full-width search visits ~10^5 nodes;
copying a 42-element list at each of them is 10^5 allocations for arithmetic
that a make/unmake pair does in place.

If you take one habit from this file, take that one: the interface a benchmark
harness uses and the interface a tree search uses are not the same interface,
and pretending otherwise is how a search ends up ten times slower than it needs
to be.

Board convention (state it once, obey it everywhere):

    index = row * COLS + col,  row 0 is the BOTTOM row (where pieces land).

Row 0 at the bottom means gravity is `heights[col] += 1`, which is one line and
cannot be got backwards. Row 0 at the top — the way you would print it — makes
gravity `ROWS - 1 - heights[col]`, which can be, and eventually is.
"""

from __future__ import annotations


import numpy as np

ROWS = 6
COLS = 7
N_CELLS = ROWS * COLS
CONNECT = 4

# Piece values. +1 / -1 rather than 1 / 2 because the whole evaluation function
# below is then a sum: a line owned by one side has |sum| == number of pieces,
# and a MIXED line — the only kind that can never be completed — is the one case
# where |sum| < number of pieces. That single identity replaces a nest of
# conditionals. See `evaluate_position`.
EMPTY = 0
YELLOW = 1   # the player who moves first
RED = -1

DRAW = 0


# ---------------------------------------------------------------------------
# Precomputed line tables. Built once at import, used millions of times.
# ---------------------------------------------------------------------------


def _all_lines() -> list[tuple[int, int, int, int]]:
    """Every set of four collinear cells on the board. 69 of them on 6x7.

    Enumerated once at import rather than recomputed per evaluation. The static
    evaluator touches every line on every leaf; at ~10^4 leaves per move that is
    ~10^6 line reads, and rebuilding the list inside the loop is the difference
    between a search that answers in a second and one that answers in a minute.
    """
    lines: list[tuple[int, int, int, int]] = []
    for r in range(ROWS):
        for c in range(COLS):
            base = r * COLS + c
            if c + CONNECT <= COLS:                      # horizontal
                lines.append(tuple(base + i for i in range(CONNECT)))  # type: ignore[arg-type]
            if r + CONNECT <= ROWS:                      # vertical
                lines.append(tuple(base + i * COLS for i in range(CONNECT)))  # type: ignore[arg-type]
            if c + CONNECT <= COLS and r + CONNECT <= ROWS:            # diagonal /
                lines.append(tuple(base + i * (COLS + 1) for i in range(CONNECT)))  # type: ignore[arg-type]
            if c - CONNECT + 1 >= 0 and r + CONNECT <= ROWS:           # diagonal \
                lines.append(tuple(base + i * (COLS - 1) for i in range(CONNECT)))  # type: ignore[arg-type]
    return lines


LINES: tuple[tuple[int, int, int, int], ...] = tuple(_all_lines())

# For terminal detection we do NOT scan all 69 lines. A win can only be created
# by the piece just played, so we scan only the lines through that cell — at
# most 13, typically 7. That is a constant-factor saving of roughly 10x on the
# single hottest operation in the entire product.
LINES_THROUGH: tuple[tuple[tuple[int, int, int, int], ...], ...] = tuple(
    tuple(line for line in LINES if cell in line) for cell in range(N_CELLS)
)

# Static evaluation weights, indexed by how many of a line's four cells one side
# already owns. Three-in-a-row with the fourth cell open is worth far more than
# three separate twos, which is why this is convex rather than linear. The
# absolute scale is arbitrary; only the ratios matter, and only up to the point
# where a real terminal score (`WIN_SCORE`) must dominate every heuristic score.
LINE_WEIGHTS = (0, 1, 12, 60)
WIN_SCORE = 100_000            # bigger than any reachable heuristic sum (69 * 60)
CENTRE_BONUS = 3               # per own piece in the centre column


class IllegalMoveError(ValueError):
    """Raised when a column is full or out of range.

    A distinct type rather than a bare ValueError because `/act` turns this into
    a 422 with the legal columns listed, and a caller who sent column 9 needs a
    different message from a caller who sent a malformed board.
    """


# ---------------------------------------------------------------------------
# Position — the search data structure
# ---------------------------------------------------------------------------


class Position:
    """A Connect Four position with make/unmake.

    Not a dataclass and not frozen, on purpose. Every search in `search/` mutates
    this object and undoes the mutation on the way back up; the invariant that
    makes that safe is that `pop()` exactly reverses `push()`, which is why
    `history` stores the column played and `winner` is recomputed rather than
    stacked. Read those five lines of `pop()` before you extend this class.
    """

    __slots__ = ("board", "heights", "player", "history", "winner", "n_pieces")

    def __init__(
        self,
        board: list[int] | None = None,
        heights: list[int] | None = None,
        player: int = YELLOW,
    ) -> None:
        self.board: list[int] = board if board is not None else [EMPTY] * N_CELLS
        self.heights: list[int] = (
            heights if heights is not None else [0] * COLS
        )
        self.player: int = player
        self.history: list[int] = []
        # Maintained incrementally by push/pop. A `sum(v != EMPTY for v in board)`
        # inside `push` would be a 42-element scan on the hottest path in the
        # product, to compute a number we already know how to update in O(1).
        self.n_pieces: int = sum(1 for v in self.board if v != EMPTY)
        self.winner: int | None = self._scan_for_winner()

    # -- construction -------------------------------------------------------

    @classmethod
    def from_board(cls, board: list[int] | np.ndarray, player: int) -> Position:
        """Rebuild a position from a flat board, validating the physics.

        This is the constructor the API uses on every request, so it is also the
        place where a hostile or simply wrong payload is rejected. It checks two
        things a naive `Position(board)` would happily accept:

          * FLOATING PIECES. Connect Four has gravity; a cell may only be
            occupied if the cell below it is. A board with a piece hovering in
            mid-air is not a position this game can reach, and searching from it
            produces confident nonsense.
          * IMPOSSIBLE PIECE COUNTS. Yellow moves first, so yellow has either
            the same number of pieces as red or exactly one more. Anything else
            means the caller has invented a position, and the move we return
            would be advice about a game nobody is playing.

        Rejecting here rather than "being permissive" is the point. A search
        service that answers questions about unreachable positions is a service
        whose answers cannot be checked against a real game.
        """
        flat = [int(v) for v in np.asarray(board).ravel()]
        if len(flat) != N_CELLS:
            raise ValueError(f"board must have {N_CELLS} cells, got {len(flat)}")
        if any(v not in (EMPTY, YELLOW, RED) for v in flat):
            raise ValueError("board cells must each be -1, 0 or +1")
        if player not in (YELLOW, RED):
            raise ValueError(f"player must be +1 or -1, got {player}")

        heights = [0] * COLS
        for c in range(COLS):
            column = [flat[r * COLS + c] for r in range(ROWS)]
            filled = 0
            for v in column:
                if v == EMPTY:
                    break
                filled += 1
            if any(v != EMPTY for v in column[filled:]):
                raise ValueError(
                    f"column {c} has a floating piece — Connect Four has gravity, "
                    "so a cell may only be occupied if the cell below it is"
                )
            heights[c] = filled

        yellow = sum(1 for v in flat if v == YELLOW)
        red = sum(1 for v in flat if v == RED)
        if yellow not in (red, red + 1):
            raise ValueError(
                f"impossible piece counts (yellow={yellow}, red={red}): yellow "
                "moves first, so yellow has either as many pieces as red or one more"
            )
        # The side to move is implied by the counts. We take it from the caller
        # anyway and CHECK it, because the caller's belief about whose turn it is
        # is exactly the thing a round trip through JSON can corrupt.
        implied = YELLOW if yellow == red else RED
        if player != implied:
            raise ValueError(
                f"player {player} contradicts the board: with yellow={yellow} and "
                f"red={red} pieces it is {implied}'s turn"
            )
        return cls(board=flat, heights=heights, player=player)

    def copy(self) -> Position:
        """A deep-enough copy: the search never needs `history`.

        Used by the tournament harness and by MCTS's simulation phase, which
        both want an independent position they can wreck. NOT used inside the
        minimax recursion — see the class docstring.
        """
        p = Position.__new__(Position)
        p.board = list(self.board)
        p.heights = list(self.heights)
        p.player = self.player
        p.history = []
        p.n_pieces = self.n_pieces
        p.winner = self.winner
        return p

    # -- moves --------------------------------------------------------------

    def legal_moves(self) -> list[int]:
        """The legal-move mask, as a list of column indices.

        Returned as columns rather than as a boolean array because every
        consumer (search ordering, MCTS child expansion, the API's move
        validation) wants to iterate them, and `np.flatnonzero(mask)` in an
        inner loop is a NumPy call per node for a seven-element list.
        `action_mask()` below returns the boolean form for the Gymnasium `info`
        dict, where the convention is a mask.
        """
        if self.winner is not None:
            return []          # a finished game has no legal continuations
        h = self.heights
        return [c for c in range(COLS) if h[c] < ROWS]

    def action_mask(self) -> np.ndarray:
        """Boolean mask over the seven columns, for `info["action_mask"]`.

        Masking is not decoration in a two-player product. An agent that can
        propose a full column will, at some point, propose one to a human on the
        other side of a browser; the mask is what turns that into an impossible
        state rather than an exception in front of a user.
        """
        mask = np.zeros(COLS, dtype=bool)
        for c in self.legal_moves():
            mask[c] = True
        return mask

    def push(self, col: int) -> None:
        """Play `col` for the side to move. O(lines through the landing cell)."""
        if not 0 <= col < COLS:
            raise IllegalMoveError(f"column {col} is outside [0, {COLS})")
        if self.winner is not None:
            raise IllegalMoveError("the game is already over")
        row = self.heights[col]
        if row >= ROWS:
            raise IllegalMoveError(
                f"column {col} is full; legal columns are {self.legal_moves()}"
            )
        idx = row * COLS + col
        self.board[idx] = self.player
        self.heights[col] = row + 1
        self.history.append(col)
        self.n_pieces += 1
        if self._wins_through(idx, self.player):
            self.winner = self.player
        elif self.n_pieces >= N_CELLS:
            self.winner = DRAW
        self.player = -self.player

    def pop(self) -> None:
        """Undo the last `push`. Exactly reverses it — check this if search breaks.

        `winner` is reset to None rather than restored from a stack because the
        only way a position can be terminal AFTER a pop is if it was terminal
        before the push, and `push` refuses to move in a terminal position. That
        argument is why there is no undo stack here; if you relax `push`, this
        line becomes a bug.
        """
        if not self.history:
            raise IndexError("pop from a position with no pushed moves")
        col = self.history.pop()
        self.heights[col] -= 1
        self.board[self.heights[col] * COLS + col] = EMPTY
        self.n_pieces -= 1
        self.winner = None
        self.player = -self.player

    def _wins_through(self, idx: int, player: int) -> bool:
        board = self.board
        for a, b, c, d in LINES_THROUGH[idx]:
            if board[a] == player and board[b] == player and board[c] == player \
                    and board[d] == player:
                return True
        return False

    def _scan_for_winner(self) -> int | None:
        """Full 69-line scan. Used ONCE, when a position arrives from outside.

        Inside the search we never call this — `push` knows which cell changed.
        Keeping the slow, obviously-correct version for the boundary and the
        fast incremental version for the hot path is the standard shape of this
        kind of optimisation, and `tests/test_env_connect_four.py` checks the
        two agree.
        """
        board = self.board
        for a, b, c, d in LINES:
            v = board[a]
            if v != EMPTY and board[b] == v and board[c] == v and board[d] == v:
                return v
        if all(v != EMPTY for v in board):
            return DRAW
        return None

    # -- terminal state -----------------------------------------------------

    def is_terminal(self) -> bool:
        return self.winner is not None

    def result_for(self, player: int) -> float:
        """+1 win, 0 draw, -1 loss, from `player`'s point of view.

        Every value in this product is expressed from a named player's point of
        view and never "from the board's". Sign errors in game search are both
        the most common bug and the hardest to see, because a sign-flipped agent
        still plays legal moves — it just plays them badly, which looks like a
        weak heuristic rather than a defect.
        """
        if self.winner is None:
            raise ValueError("result_for called on a non-terminal position")
        if self.winner == DRAW:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    # -- rendering ----------------------------------------------------------

    def to_grid(self) -> list[list[int]]:
        """Row-major, TOP row first — the order a human reads a board in.

        Note that this reverses the internal order. The UI wants the top row
        first; the search wants the bottom row first. Converting at the boundary
        keeps exactly one flip in the codebase instead of one per consumer.
        """
        return [
            [self.board[r * COLS + c] for c in range(COLS)]
            for r in range(ROWS - 1, -1, -1)
        ]

    def __str__(self) -> str:
        glyph = {EMPTY: ".", YELLOW: "Y", RED: "R"}
        rows = "\n".join(" ".join(glyph[v] for v in row) for row in self.to_grid())
        return rows + "\n" + " ".join(str(c) for c in range(COLS))


# ---------------------------------------------------------------------------
# The wire format. This is what survives the JSON round trip.
# ---------------------------------------------------------------------------
#
# `/act` takes `state: list[float]`, the same field every other product in this
# course uses, so the encoding has to fit in a flat list of numbers. It is
# 43 long: 42 cells then the side to move.
#
# Why floats when every value is an integer? Because JSON has one number type
# and Pydantic will hand back `1.0` for a value that was written as `1`. Encoding
# to float deliberately and decoding with a tolerance check means the round trip
# is tested rather than assumed — `tests/test_env_connect_four.py` asserts
# `decode(encode(p))` reproduces the position exactly, which is the property the
# whole service depends on and the one that silently breaks when someone
# "tidies" this to a nested list.

STATE_DIM = N_CELLS + 1


def encode_state(position: Position) -> list[float]:
    """Position -> the 43-float wire encoding."""
    return [float(v) for v in position.board] + [float(position.player)]


def decode_state(state: list[float] | np.ndarray) -> Position:
    """The 43-float wire encoding -> a validated Position.

    Raises ValueError with a readable message on anything that is not a legal
    Connect Four position. `/act` turns that into a 422; see api/main.py.
    """
    flat = np.asarray(state, dtype=np.float64).ravel()
    if flat.size != STATE_DIM:
        raise ValueError(
            f"a Connect Four state is {STATE_DIM} numbers "
            f"({N_CELLS} cells then the side to move), got {flat.size}"
        )
    # Round, then check the rounding did not move anything. A caller who sends
    # 0.5 has a bug, and silently reading it as 0 or 1 hides their bug inside
    # our answer.
    rounded = np.rint(flat)
    if not np.allclose(flat, rounded, atol=1e-6):
        raise ValueError("state must contain integers (-1, 0, +1); got fractional values")
    cells = [int(v) for v in rounded[:N_CELLS]]
    player = int(rounded[N_CELLS])
    return Position.from_board(cells, player)


# ---------------------------------------------------------------------------
# Static evaluation — the "domain evaluation function" the heuristic ordering
# and the depth-limited search both consume.
# ---------------------------------------------------------------------------


def evaluate_position(position: Position, player: int) -> float:
    """Heuristic score of `position` from `player`'s point of view.

    The identity that makes this cheap: cells are -1, 0 and +1, so for the four
    cells of a line, `s = sum` and `n = count of non-empty`. The line is owned
    by exactly one side precisely when `abs(s) == n`, and then `n` is that
    side's piece count. A mixed line — one that neither side can ever complete,
    and therefore contributes nothing — is the only case where `abs(s) < n`.
    One subtraction replaces the branch nest you would otherwise write, and this
    function runs at every leaf of every search.

    The centre bonus encodes the one piece of Connect Four theory worth
    hard-coding: the centre column participates in more lines than any other, so
    a piece there is worth more than a piece on the edge. Without it, a
    shallow search opens on column 0 and never recovers.

    This is a HEURISTIC. It is not a value function, it was not learned, and it
    is wrong in positions with forced sequences longer than the search horizon.
    That is exactly what the AlphaZero-inspired agent replaces, and the
    benchmark is what tells you whether replacing it helped.
    """
    board = position.board
    if position.winner is not None:
        return WIN_SCORE * position.result_for(player)

    score = 0
    for a, b, c, d in LINES:
        va, vb, vc, vd = board[a], board[b], board[c], board[d]
        s = va + vb + vc + vd
        if s == 0:
            continue                     # empty line, or perfectly mixed: worth nothing
        n = (va != 0) + (vb != 0) + (vc != 0) + (vd != 0)
        if s == n:
            score += LINE_WEIGHTS[n]
        elif s == -n:
            score -= LINE_WEIGHTS[n]
    centre = COLS // 2
    for r in range(ROWS):
        v = board[r * COLS + centre]
        if v != EMPTY:
            score += CENTRE_BONUS * v
    return float(score * player)


def winning_moves(position: Position, player: int) -> list[int]:
    """Columns that complete a four-in-a-row for `player` immediately.

    Used by the revised MCTS agent's playout policy and by the tactical check in
    `search/agents.py`. Separated out because "can I win right now, and can my
    opponent" is the single cheapest piece of lookahead in the game, and an
    agent that ignores it looks obviously broken to a human opponent regardless
    of how principled its search is.
    """
    out = []
    for col in position.legal_moves():
        row = position.heights[col]
        idx = row * COLS + col
        position.board[idx] = player
        if position._wins_through(idx, player):
            out.append(col)
        position.board[idx] = EMPTY
    return out
