"""
api/policy.py — the entire serving-side inference stack.

Read the imports. There are three, and none of them is a machine-learning
framework. This is the file that makes the no-torch rule cost nothing: a
trained policy in this course is either a lookup table or two or three matrix
multiplies, and both are a handful of lines of NumPy.

What lives here:
  * `PolicyArtifactStore` — discovers `.npz` archives in POLICY_DIR, checksums
    them, and keeps them loaded.
  * `TabularPolicy`  — Topics 1, 2 and 5: argmax over a row of a Q-table.
  * `MLPPolicy`      — Topics 3, 4 and 6: ReLU stack, then softmax (discrete)
                       or a tanh-squashed Gaussian mean (continuous).

What does NOT live here: any training loop, any optimiser, any gradient.
If you find yourself importing something to fit a parameter, you are in the
wrong directory — that belongs in train/.
"""

from __future__ import annotations

import hashlib
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Forward passes. Written out longhand on purpose: inference is arithmetic.
# ---------------------------------------------------------------------------


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax.

    Subtracting the max is not a nicety. Without it, a logit of 800 — which a
    diverging policy will produce — overflows to inf and the whole vector
    becomes NaN, and the service returns 500 on an input the training run
    never saw.
    """
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


@dataclass
class TabularPolicy:
    """Q-table of shape (n_states, n_actions).

    `V` and `pi` are optional companions written by `train/export.py`. They are
    carried on the policy object rather than re-derived because for the Monte
    Carlo artifact V is NOT `Q.max(axis=1)` — it is an independent first-visit
    evaluation, and the whole Value Map tab is about the difference between
    those two things. Deriving it here would quietly replace the measured
    number with the optimised one.
    """

    Q: np.ndarray
    V: np.ndarray | None = None
    pi: np.ndarray | None = None

    kind = "tabular"

    @property
    def obs_dim(self) -> int:
        return 1  # a single state index

    @property
    def n_actions(self) -> int:
        return int(self.Q.shape[1])

    def act(self, state: np.ndarray, deterministic: bool = True) -> tuple[int, float]:
        idx = int(round(float(state[0])))
        if not 0 <= idx < self.Q.shape[0]:
            raise ValueError(
                f"state index {idx} outside [0, {self.Q.shape[0]}) for this artifact"
            )
        row = self.Q[idx]
        if deterministic:
            return int(np.argmax(row)), float(np.max(row))
        p = softmax(row)
        return int(np.random.choice(len(p), p=p)), float(np.max(row))


@dataclass
class MLPPolicy:
    """Feed-forward policy exported from PyTorch.

    Expected archive keys: W0, b0, W1, b1, ... in order, plus an optional
    `log_std` for a continuous (SAC-style) actor and optional `obs_mean` /
    `obs_std` for observation normalisation.

    The weight convention is `h = W @ h + b` (PyTorch's nn.Linear stores
    weight as (out, in), so export it WITHOUT transposing and this works). If
    the equivalence test in tests/ fails, this convention is the first thing
    to check — a transposed matrix or a dropped bias accounts for nearly every
    failure of that test.
    """

    layers: list[tuple[np.ndarray, np.ndarray]]
    discrete: bool = True
    log_std: np.ndarray | None = None
    obs_mean: np.ndarray | None = None
    obs_std: np.ndarray | None = None
    squash: bool = False  # tanh squashing, as SAC uses

    kind = "mlp"

    @property
    def obs_dim(self) -> int:
        return int(self.layers[0][0].shape[1])

    @property
    def n_actions(self) -> int:
        return int(self.layers[-1][0].shape[0])

    def _trunk(self, x: np.ndarray) -> np.ndarray:
        if self.obs_mean is not None and self.obs_std is not None:
            x = (x - self.obs_mean) / (self.obs_std + 1e-8)
        h = x
        for i, (W, b) in enumerate(self.layers):
            h = W @ h + b
            if i < len(self.layers) - 1:      # no activation on the output layer
                h = relu(h)
        return h

    def act(self, state: np.ndarray, deterministic: bool = True):
        x = np.asarray(state, dtype=np.float64)
        if x.shape[0] != self.obs_dim:
            raise ValueError(
                f"observation has dimension {x.shape[0]}, artifact expects {self.obs_dim}"
            )
        out = self._trunk(x)
        if self.discrete:
            p = softmax(out)
            if deterministic:
                return int(np.argmax(p)), float(np.max(p))
            return int(np.random.choice(len(p), p=p)), float(np.max(p))

        # Continuous: `out` is the pre-squash mean. A deterministic action is
        # tanh(mean) — note that this is NOT the mean of the squashed
        # distribution, which has no closed form. Say so in your model card.
        mean = out
        if not deterministic and self.log_std is not None:
            mean = mean + np.exp(self.log_std) * np.random.randn(*mean.shape)
        a = np.tanh(mean) if self.squash else mean
        return a.tolist(), float(np.linalg.norm(mean))


# ---------------------------------------------------------------------------
# Artifact discovery and loading
# ---------------------------------------------------------------------------


def sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class PolicyArtifactStore:
    """Loads every `.npz` in POLICY_DIR once, and remembers its checksum.

    The checksum is not decoration. `/act` returns it and `audit_log` stores
    it, so six weeks later you can answer "which artifact produced this
    action" with a join rather than with a guess.
    """

    def __init__(
        self,
        policy_dir: str | pathlib.Path = "policies",
        default_name: str | None = None,
    ) -> None:
        self.dir = pathlib.Path(policy_dir)
        # Which artifact "default" resolves to when several are registered.
        # The base template's rule — resolve only when there is exactly one —
        # stops working the moment a topic ships two agents, and the failure is
        # a 404 on a request that names nothing wrong. Naming the default is a
        # product decision as much as a technical one: this topic points it at
        # the EXACT solution, so a caller who does not choose gets the answer
        # that is provably right rather than the one that was learned.
        self.default_name = default_name
        self._cache: dict[str, tuple[Any, dict[str, Any]]] = {}
        self.reload()

    def reload(self) -> None:
        self._cache.clear()
        if not self.dir.exists():
            return
        for path in sorted(self.dir.glob("*.npz")):
            try:
                policy, meta = self._load_one(path)
            except Exception as exc:  # a corrupt artifact must not kill the app
                meta = {"name": path.stem, "error": str(exc)}
                policy = None
            self._cache[path.stem] = (policy, meta)
        # "default" resolves to the configured artifact if it loaded, otherwise
        # to the single artifact if there is exactly one — so a fresh clone
        # with one policy still needs no configuration.
        if "default" not in self._cache:
            if self.default_name and self.default_name in self._cache:
                self._cache["default"] = self._cache[self.default_name]
            elif len(self._cache) == 1:
                self._cache["default"] = next(iter(self._cache.values()))

    def _load_one(self, path: pathlib.Path):
        z = np.load(path, allow_pickle=False)
        keys = set(z.files)
        meta: dict[str, Any] = {
            "name": path.stem,
            "format": "npz",
            "bytes": path.stat().st_size,
            "sha256": sha256_of(path),
        }
        if "Q" in keys:
            pol = TabularPolicy(
                Q=np.asarray(z["Q"], dtype=np.float64),
                V=np.asarray(z["V"], dtype=np.float64) if "V" in keys else None,
                # Stored as float32 like every other array in the archive, so
                # cast back to int here rather than indexing a table with a
                # float and getting a TypeError at the first request.
                pi=np.asarray(z["pi"], dtype=np.int64) if "pi" in keys else None,
            )
            meta |= {"kind": "tabular", "obs_dim": 1, "n_actions": pol.n_actions}
            return pol, meta

        layers = []
        i = 0
        while f"W{i}" in keys and f"b{i}" in keys:
            layers.append((np.asarray(z[f"W{i}"], dtype=np.float64),
                           np.asarray(z[f"b{i}"], dtype=np.float64)))
            i += 1
        if not layers:
            raise ValueError(
                f"{path.name}: no 'Q' and no 'W0'/'b0' — see train/export.py for the layout"
            )
        pol = MLPPolicy(
            layers=layers,
            discrete=("log_std" not in keys),
            log_std=np.asarray(z["log_std"], dtype=np.float64) if "log_std" in keys else None,
            obs_mean=np.asarray(z["obs_mean"], dtype=np.float64) if "obs_mean" in keys else None,
            obs_std=np.asarray(z["obs_std"], dtype=np.float64) if "obs_std" in keys else None,
            squash=bool("log_std" in keys),
        )
        meta |= {"kind": "mlp", "obs_dim": pol.obs_dim, "n_actions": pol.n_actions}
        return pol, meta

    # -- accessors ----------------------------------------------------------

    def names(self) -> list[str]:
        return [n for n in self._cache if n != "default"]

    def metadata(self) -> list[dict[str, Any]]:
        seen, out = set(), []
        for name, (_, meta) in self._cache.items():
            if meta.get("sha256") in seen:
                continue
            seen.add(meta.get("sha256"))
            out.append(meta)
        return out

    def get(self, name: str):
        if name not in self._cache:
            raise KeyError(name)
        policy, meta = self._cache[name]
        if policy is None:
            raise ValueError(meta.get("error", "artifact failed to load"))
        return policy, meta

    def loaded(self) -> bool:
        return any(p is not None for p, _ in self._cache.values())
