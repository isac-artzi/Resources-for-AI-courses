"""
search/net.py — the learned evaluator, evaluated in NumPy.

This is the serving half of the AlphaZero-inspired agent. The other half —
self-play, the loss, the optimiser — lives in `train/selfplay.py` and imports
torch. Nothing in this file does, and `tests/test_no_torch.py` is what keeps it
that way.

Look at how little there is here. A policy-value network for Connect Four is a
two-layer trunk and two linear heads: four matrix multiplies, a ReLU, a softmax
and a tanh. That is the entire "AI" of the strongest agent in this product at
inference time. The 490 MB framework is for computing the gradients, and the
gradients were computed last week on a laptop.

Archive layout — `train/selfplay.py` writes it and this file reads it, so if you
change one, change the other in the same commit:

    W0, b0, W1, b1   the shared trunk (ReLU between layers)
    Wp, bp           policy head  -> 7 logits, one per column
    Wv, bv           value head   -> 1 scalar, tanh-squashed to [-1, 1]

The input is 84 floats: two 42-cell binary planes, "my pieces" then "the
opponent's", from the point of view of the SIDE TO MOVE. See
`shared/preprocess.py::canonical_planes` for why it is canonicalised rather than
fed the raw board plus a turn indicator.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np

from envs.connect_four import COLS, Position
from shared.preprocess import canonical_planes


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax. See api/policy.py for why the max subtraction
    is load-bearing rather than tidy."""
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


@dataclass
class PolicyValueNet:
    """Two heads on a shared trunk, evaluated with NumPy.

    Kept separate from `api/policy.py`'s `MLPPolicy` even though the trunk is
    identical, because the contract is different: `MLPPolicy.act()` answers
    "which action", while this answers "what is the prior over actions AND what
    is this position worth". Folding a second head into `MLPPolicy` would make
    every other topic's policy carry a value head it does not have.
    """

    W0: np.ndarray
    b0: np.ndarray
    W1: np.ndarray
    b1: np.ndarray
    Wp: np.ndarray
    bp: np.ndarray
    Wv: np.ndarray
    bv: np.ndarray

    kind = "value-net"

    @property
    def obs_dim(self) -> int:
        return int(self.W0.shape[1])

    @property
    def n_actions(self) -> int:
        return int(self.Wp.shape[0])

    @classmethod
    def from_npz(cls, path: str | pathlib.Path) -> "PolicyValueNet":
        """Load from the exported archive. `allow_pickle=False`, always.

        A pickle is arbitrary code execution dressed as a data file. This
        service loads artifacts from a directory that a deployment process
        writes to; a pickled artifact would mean anyone who can write there can
        run code in the service. `.npz` of plain float arrays cannot.
        """
        z = np.load(pathlib.Path(path), allow_pickle=False)
        missing = [k for k in ("W0", "b0", "W1", "b1", "Wp", "bp", "Wv", "bv")
                   if k not in z.files]
        if missing:
            raise ValueError(
                f"{path}: missing {missing} — this is not a policy-value archive. "
                "Run `python -m train.train` to produce one."
            )
        return cls(**{k: np.asarray(z[k], dtype=np.float64)
                      for k in ("W0", "b0", "W1", "b1", "Wp", "bp", "Wv", "bv")})

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """(logits over columns, value in [-1, 1]) for one canonical observation."""
        h = _relu(self.W0 @ x + self.b0)
        h = _relu(self.W1 @ h + self.b1)
        logits = self.Wp @ h + self.bp
        value = float(np.tanh(self.Wv @ h + self.bv)[0])
        return logits, value

    def act(self, state: np.ndarray, deterministic: bool = True) -> tuple[int, float]:
        """The `Policy` protocol `api/policy.py` expects, so `/act` can serve
        this artifact by name like any other.

        It is the network's RAW PRIOR — one forward pass, no search — and that
        is worth being explicit about, because it is the honest ablation: the
        difference between this and `agent: "alphazero"` is exactly what the
        200 simulations of PUCT are buying. Serving it is how you measure that
        difference rather than assert it.

        Note it CANNOT mask illegal columns: it is handed an observation vector,
        not a position, and a full column is not recoverable from the canonical
        planes alone... except that it is — a column is full when all six of its
        cells are occupied in one plane or the other. Doing that reconstruction
        here would put board logic in the artifact loader, which is where it
        does not belong. Callers who need masking name the agent instead; the
        `/act` search path decodes a real `Position` and masks properly.
        """
        x = np.asarray(state, dtype=np.float64).ravel()
        if x.shape[0] != self.obs_dim:
            raise ValueError(
                f"observation has dimension {x.shape[0]}, artifact expects "
                f"{self.obs_dim} (two 42-cell planes — see shared/preprocess.py)"
            )
        logits, value = self.forward(x)
        p = _softmax(logits)
        if deterministic:
            return int(np.argmax(p)), value
        return int(np.random.choice(len(p), p=p)), value

    def evaluate(self, position: Position) -> tuple[np.ndarray, float]:
        """(prior over the 7 columns, value) for a position, ready for PUCT.

        Two things happen here that are easy to get wrong and expensive to
        debug:

        1. **The prior is masked to legal moves and renormalised.** An unmasked
           softmax puts probability on full columns, and PUCT will happily spend
           its exploration budget on children that cannot exist. Masking after
           the softmax rather than setting logits to -inf before it is the same
           thing numerically here and avoids an inf/nan class of bug.
        2. **The value is from the side to move's point of view**, because the
           input planes were canonicalised that way. `search/mcts.py` backs it up
           the tree assuming exactly that. If you change the encoding, this
           sentence is the one that stops being true.
        """
        logits, value = self.forward(canonical_planes(position.board, position.player))
        prior = _softmax(logits)
        mask = np.zeros(COLS, dtype=np.float64)
        for col in position.legal_moves():
            mask[col] = 1.0
        prior = prior * mask
        total = prior.sum()
        if total <= 0.0:
            # The network put all its mass on illegal columns — possible early
            # in training. Fall back to uniform over the legal moves rather than
            # dividing by zero and poisoning the whole search with NaNs.
            prior = mask / max(mask.sum(), 1.0)
        else:
            prior = prior / total
        return prior, value

    def value_of(self, position: Position, player: int) -> float:
        """Static evaluation in the units `search/minimax.py` expects.

        Lets the learned network be dropped into the alpha-beta scaffold in
        place of the hand-written heuristic — one line at the call site, and a
        directly comparable node count. The scaling to the heuristic's range is
        arbitrary but must be consistent: a value function whose outputs are in
        [-1, 1] mixed with terminal scores of 100,000 would make every
        non-terminal position indistinguishable from every other.
        """
        _, value = self.evaluate(position)
        sign = 1.0 if position.player == player else -1.0
        return sign * value * 1000.0
