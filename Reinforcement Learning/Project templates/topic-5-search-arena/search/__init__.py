"""
search/ — every decision procedure in the product, in one place, importing NumPy
and nothing heavier.

This package is on the SERVING path. `api/main.py` imports it directly, so
nothing here may import torch, gymnasium, pandas or a training module. The
learned evaluator is loaded from a `.npz` and evaluated with three matrix
multiplies in `search/net.py`; the code that fits those matrices lives in
`train/selfplay.py` and is never imported here. `tests/test_no_torch.py` is what
holds that line.

    minimax.py   ONE recursion. Exhaustive and heuristic search are the same
                 function with a different node-ordering callback; alpha-beta is
                 a flag on it. Read this file first.
    ordering.py  the ordering callbacks, which is where the two baseline
                 strategies actually differ.
    mcts.py      Monte Carlo tree search with UCT, and the PUCT variant the
                 learned agent uses.
    net.py       NumPy forward pass for the exported policy-value network.
    agents.py    the registry: name -> configured agent. The API, the UI and the
                 benchmark harness all resolve agents through this one table.
    budget.py    the node budget, which is a hard bound and not a suggestion.
"""

from __future__ import annotations

from search.budget import NodeBudget, NodeBudgetExceeded
from search.minimax import SearchStats, search_root
from search.ordering import heuristic_ordering, make_beam_ordering, natural_ordering

__all__ = [
    "NodeBudget",
    "NodeBudgetExceeded",
    "SearchStats",
    "heuristic_ordering",
    "make_beam_ordering",
    "natural_ordering",
    "search_root",
]
