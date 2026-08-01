"""
search/budget.py — the node budget, which is the only thing standing between a
depth-9 request and a dead free-tier instance.

The problem this solves is specific and it will happen to you. Connect Four has
a branching factor of 7, so a full-width search at depth d visits about 7^d
nodes: 117,649 at depth 6 and 5.7 million at depth 8. A user (or a crawler, or
your own Streamlit app with a slider) sends `depth: 9` and the request takes
minutes, holds a worker, and the container is killed for exceeding its memory or
its request timeout. Nothing about the search is wrong; it was simply asked a
question too large to answer.

Three properties this design insists on, each because the obvious alternative is
worse:

  1. **The budget is counted, not estimated.** A depth cap is not a budget:
     alpha-beta at depth 6 might visit 2,000 nodes or 60,000 depending entirely
     on move ordering, and a bound that varies by 30x is not a bound.

  2. **Exhaustion degrades, it does not raise — at the top level.** When the
     budget runs out, the search stops descending and scores the remaining
     positions statically. The caller gets a legal move and a flag saying the
     answer was truncated. A 500 in the middle of a human's game is a worse
     outcome than a slightly weaker move, and "we ran out of budget" is
     information the caller can act on; a traceback is not.

  3. **The consumption is reported.** `/act` and `/game` return
     `nodes_expanded` alongside the move, so the number in the UI and the number
     in the `games` table are the same number, produced by the same counter.
     A budget you cannot observe is a budget you cannot tune.

`NodeBudgetExceeded` exists for the ONE caller that wants the strict behaviour:
the scalability probe in `train/benchmark.py`, which needs to know that a
configuration did not fit rather than to receive its best guess.
"""

from __future__ import annotations


class NodeBudgetExceeded(RuntimeError):
    """Raised only when a caller asked for strict enforcement. See NodeBudget."""

    def __init__(self, limit: int) -> None:
        super().__init__(
            f"search exceeded its node budget of {limit:,} nodes. "
            "Reduce the depth or raise the budget — see search/budget.py for why "
            "this bound exists at all."
        )
        self.limit = limit


class NodeBudget:
    """A counter with a ceiling, threaded through every search in this package.

    Deliberately a mutable object passed by reference rather than an integer
    returned up the call stack. A recursion that has to thread a running total
    through its return value ends up with every function returning a tuple whose
    second element nobody reads, and the first time someone forgets to add a
    child's count the budget silently stops working. One object, one `spend()`,
    one place to audit.
    """

    __slots__ = ("limit", "spent", "strict", "exhausted")

    def __init__(self, limit: int, strict: bool = False) -> None:
        if limit < 1:
            raise ValueError("a node budget below 1 cannot evaluate the root")
        self.limit = int(limit)
        self.spent = 0
        self.strict = bool(strict)
        # Sticky: once a search has been truncated, every answer derived from it
        # is a truncated answer, and clearing this per-branch would let a report
        # claim a complete search that was not one.
        self.exhausted = False

    def spend(self, n: int = 1) -> bool:
        """Charge `n` nodes. Returns False once the budget is gone.

        Callers check the return value and stop descending; they do not have to
        remember to check `exhausted` as well.
        """
        self.spent += n
        if self.spent >= self.limit:
            self.exhausted = True
            if self.strict:
                raise NodeBudgetExceeded(self.limit)
            return False
        return True

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.spent)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return f"NodeBudget(spent={self.spent}, limit={self.limit}, exhausted={self.exhausted})"
