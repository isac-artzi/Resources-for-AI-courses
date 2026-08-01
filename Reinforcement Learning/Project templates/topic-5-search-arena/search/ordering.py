"""
search/ordering.py — the node-ordering callbacks.

This tiny module is the entire difference between "exhaustive search" and
"heuristic search" in this product. The syllabus asks for one scaffold with two
behaviours rather than two implementations, and this is where that requirement
is discharged: `search/minimax.py` has one recursion, and it calls one of these
functions to decide which children to visit and in what order.

The callback signature is:

    ordering(position, moves, depth_remaining) -> list[int]

and it is allowed to do TWO things:

  * REORDER the moves. This changes nothing about which nodes exist and
    everything about how many alpha-beta has to look at. It cannot change the
    value returned.
  * DROP moves (`make_beam_ordering`). This is forward pruning. It reduces the
    node count without alpha-beta's help — and it is UNSOUND: the move it drops
    might have been the winning one. The exhaustive search is a proof; a beam
    search is a guess with a smaller bill.

That distinction is the one students most often blur, so the code makes it
structural: `natural_ordering` and `heuristic_ordering` both return a
permutation of their input and are therefore value-preserving;
`make_beam_ordering` returns a subset and is not. `tests/test_alpha_beta.py`
asserts the first property; `tests/test_forced_win.py` shows the second failing
on purpose.

**The thing to understand before you report node counts.** Reordering alone,
with alpha-beta OFF, changes nothing at all: a full-width search visits every
node regardless of the order it visits them in. Ordering only pays when there
is a cutoff to trigger, which is why the node-count table in the README has four
rows and not two. Good ordering is not a search strategy; it is a multiplier on
alpha-beta.
"""

from __future__ import annotations

from typing import Callable, Protocol

from envs.connect_four import COLS, Position, evaluate_position, winning_moves

# The callback type. Named rather than inlined so that `search_root`'s signature
# reads as "inject an ordering" instead of as a wall of Callable[...].
class OrderingFn(Protocol):
    def __call__(
        self, position: Position, moves: list[int], depth_remaining: int
    ) -> list[int]: ...


# Centre-out column order: 3, 2, 4, 1, 5, 0, 6 on a 7-wide board.
# Even the "exhaustive" baseline uses index order, NOT this — see below.
_CENTRE_OUT = sorted(range(COLS), key=lambda c: abs(c - COLS // 2))


def natural_ordering(position: Position, moves: list[int], depth_remaining: int) -> list[int]:
    """Left to right, as `legal_moves()` produced them. The exhaustive baseline.

    This is deliberately the DUMBEST possible ordering, and it is the baseline
    the heuristic variant is measured against. It would be easy to make the
    baseline centre-out — it is one line and it is strictly better — but then
    the reported improvement from "heuristic ordering" would be an improvement
    over an ordering that was already half-heuristic, and the comparison would
    flatter the result. Baselines are supposed to be honest, not competitive.
    """
    return moves


def heuristic_ordering(
    position: Position, moves: list[int], depth_remaining: int
) -> list[int]:
    """Best-looking child first, by the domain evaluation function.

    The ordering is: immediate wins, then immediate blocks, then descending
    static evaluation of the resulting position, with centre-out as the
    tie-break.

    Why the tactical checks come first rather than falling out of the static
    evaluation: `evaluate_position` scores a completed four-in-a-row as
    `WIN_SCORE`, so a winning move DOES sort first on evaluation alone. The
    explicit check is not for correctness, it is for cost — `winning_moves` is a
    handful of array writes, while evaluating every child means seven full
    69-line sweeps. At the top of a search tree that is free; at depth 6 with a
    hundred thousand nodes it is most of the runtime.

    Why blocks matter for ORDERING and not just for play: if the opponent has a
    winning reply, every child except the block has the same value (lost), and
    trying the block first is what lets alpha-beta refute the other six with one
    node each.

    This function is value-preserving — it returns a permutation of `moves` — so
    a search using it returns exactly what the exhaustive search returns. Only
    the node count changes. That is the claim `tests/test_alpha_beta.py` checks.
    """
    if len(moves) <= 1:
        return moves

    me = position.player
    wins = set(winning_moves(position, me))
    blocks = set(winning_moves(position, -me))

    scored: list[tuple[int, int, float, int]] = []
    for col in moves:
        # Rank tier: 0 = wins now, 1 = must block, 2 = everything else. Sorting
        # on the tier first means we never pay for a static evaluation to
        # discover something `winning_moves` already told us.
        tier = 0 if col in wins else (1 if col in blocks else 2)
        if tier == 2:
            position.push(col)
            value = -evaluate_position(position, position.player)   # from `me`'s view
            position.pop()
        else:
            value = 0.0
        scored.append((tier, _CENTRE_OUT.index(col), value, col))

    # Sort key: tier ascending, then value DESCENDING (best first), then
    # centre-out. The negation is on `value` rather than reversing the whole
    # sort, because reversing would also invert the tier order — a classic
    # off-by-one-sort bug that produces an agent which plays its worst move.
    scored.sort(key=lambda t: (t[0], -t[2], t[1]))
    return [t[3] for t in scored]


def make_beam_ordering(width: int, inner: OrderingFn = heuristic_ordering) -> OrderingFn:
    """An ordering callback that also DROPS moves — forward pruning.

    Keeps the best `width` children by `inner`'s ranking and discards the rest.
    This is what the syllabus's discussion question means by "a heuristic search
    that explores only 30% of the nodes at each level": with b = 7 and width = 3,
    the node count at depth d falls from 7^d to 3^d, which at depth 6 is 117,649
    against 729.

    It is also, unlike everything else in this module, **unsound**. The move it
    discards may be the only winning one, and no amount of extra depth recovers
    it because the branch is gone. Use it when the alternative is not searching
    at all — and say so in the write-up rather than quoting the node saving on
    its own. A cheaper wrong answer is not an optimisation.

    The width is a property of the CALLBACK, not of the recursion. That is why
    this is a factory: `search/minimax.py` never learns that forward pruning
    exists, and adding a different pruning scheme means adding a function here
    and changing nothing else.

    The beam applies AT THE ROOT as well as inside the tree. That is what
    forward pruning means, and it has a consequence worth stating out loud: a
    beam agent can never play a move that `inner` ranked outside the top
    `width`, no matter how good that move turns out to be. If that is not the
    behaviour you want, you want a deeper search, not a wider beam.
    """
    if width < 1:
        raise ValueError("beam width must be at least 1")

    def ordering(position: Position, moves: list[int], depth_remaining: int) -> list[int]:
        return inner(position, moves, depth_remaining)[:width]

    ordering.__name__ = f"beam{width}_ordering"
    return ordering


ORDERINGS: dict[str, Callable[..., list[int]]] = {
    "natural": natural_ordering,
    "heuristic": heuristic_ordering,
}
