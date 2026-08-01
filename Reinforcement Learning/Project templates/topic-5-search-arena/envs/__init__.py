"""
envs/ — the environment, as a proper Gymnasium environment.

This topic's environment is Connect Four (6x7). The syllabus permits
Tic-Tac-Toe, and Tic-Tac-Toe is easier to get right — but it is also solved by
exhaustive search from the empty board in well under a million nodes, which
means the scalability study in this product would have nothing to measure.
Connect Four's game tree is roughly 4.5 x 10^12 positions; exhaustive search is
hopeless past about depth 8 on a laptop, heuristic ordering visibly changes the
node count, and MCTS has a reason to exist. Pick the game that makes your
measurement interesting, not the one that makes your code short.

    from envs import make_env
    env = make_env()

`api/main.py` imports THIS name and nothing deeper, so a student who swaps in
Tic-Tac-Toe or a puzzle solver replaces this module and `envs/connect_four.py`
and touches nothing in the service tier.

Two constraints that are not negotiable in this course:

  * Install only the extras you need — `gymnasium[classic-control]`, never
    `gymnasium[all]`, which still declares the obsolete swig-dependent Box2D
    package and fails to install.
  * Anything the DEPLOYED service instantiates must be pure Python or
    classic-control. Connect Four here is pure Python and NumPy.

THE IMPORT GRAPH IS PART OF THE DESIGN. `envs/connect_four.py` holds the rules
and imports nothing but NumPy. `envs/gym_env.py` holds the Gymnasium wrapper and
is imported ONLY inside `make_env()`. So `import api.main` — which needs the
rules and the search but never constructs an environment — does not load
gymnasium at all, and CI proves it by uninstalling gymnasium and importing
`search.agents`. Re-exporting `ConnectFourEnv` at the top of this file would
undo that in one line, which is why it is behind `__getattr__` below instead.
"""

from __future__ import annotations

from typing import Any

from envs.connect_four import (
    COLS,
    DRAW,
    EMPTY,
    N_CELLS,
    RED,
    ROWS,
    STATE_DIM,
    YELLOW,
    IllegalMoveError,
    Position,
    decode_state,
    encode_state,
    evaluate_position,
    winning_moves,
)

ENV_ID = "ConnectFour-6x7-v1"

__all__ = [
    "COLS",
    "DRAW",
    "EMPTY",
    "ENV_ID",
    "N_CELLS",
    "RED",
    "ROWS",
    "STATE_DIM",
    "YELLOW",
    "ConnectFourEnv",
    "IllegalMoveError",
    "Position",
    "decode_state",
    "encode_state",
    "evaluate_position",
    "make_env",
    "winning_moves",
]


def make_env(**kwargs):
    """The environment the standing endpoints and the training tier use.

    Returns the ALTERNATING-MOVE environment: one `step()` plays one move, and
    the reward is from the point of view of the player who just moved. Read the
    `ConnectFourEnv` docstring before you use it for learning — the sign
    convention is the thing that catches people out.

    Note there is no `TimeLimit` wrapper. Connect Four terminates on its own
    after at most 42 moves, so a step limit would be a second, weaker statement
    of a bound the game already guarantees, and a truncation that can never fire
    is a branch that never gets tested.
    """
    # Imported HERE, not at module scope. See the module docstring: this one
    # line is what keeps gymnasium out of the service tier's import graph.
    from envs.gym_env import ConnectFourEnv

    return ConnectFourEnv(**kwargs)


def __getattr__(name: str) -> Any:
    """Lazily expose `ConnectFourEnv` without importing gymnasium on `import envs`.

    PEP 562 module-level `__getattr__`. `from envs import ConnectFourEnv` still
    works for a test or a notebook that wants the class directly, and it pays
    the gymnasium import only at that moment. Listing it in the `from
    envs.connect_four import ...` block above would have imported gymnasium for
    every caller, including `api/main.py`, which never uses it.
    """
    if name == "ConnectFourEnv":
        from envs.gym_env import ConnectFourEnv

        return ConnectFourEnv
    raise AttributeError(f"module 'envs' has no attribute {name!r}")
