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

The arithmetic itself moved into `api/forward.py` in this topic. It is the same
arithmetic; it is in its own module so that the required NumPy/PyTorch
equivalence test can exercise the exact functions the service calls rather than
a copy of them that could drift.

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

from api.forward import (
    action_probabilities,
    check_layer_shapes,
    layers_from_npz,
    mlp_forward,
    relu,
    softmax,
)

# Re-exported so that `from api.policy import softmax` keeps working for code
# written against the base template. One definition, two names.
__all__ = [
    "MLPPolicy",
    "PolicyArtifactStore",
    "TabularPolicy",
    "relu",
    "sha256_of",
    "softmax",
]


@dataclass
class TabularPolicy:
    """Q-table of shape (n_states, n_actions)."""

    Q: np.ndarray

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
        return mlp_forward(self.layers, self._normalise(x))

    def probabilities(self, state: np.ndarray) -> np.ndarray:
        """π(·|s) as a vector. The equivalence test compares this, not the action.

        Exposed as its own method because a *distribution* is what you can
        compare against PyTorch to a tolerance. Comparing sampled actions
        compares two random number generators, and comparing argmaxes passes
        for a badly wrong network whenever the error does not happen to flip
        the winner — which, with two actions, is most of the time.
        """
        if not self.discrete:
            raise ValueError("probabilities() is only defined for a discrete policy")
        return action_probabilities(self.layers, self._normalise(state))

    def _normalise(self, x: np.ndarray) -> np.ndarray:
        """Apply the observation statistics computed at TRAINING time.

        They travel inside the archive precisely so this cannot drift.
        Recomputing them from live traffic is the classic train/serve skew bug
        — see the note in shared/preprocess.py.
        """
        x = np.asarray(x, dtype=np.float64)
        if self.obs_mean is not None and self.obs_std is not None:
            x = (x - self.obs_mean) / (self.obs_std + 1e-8)
        return x

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
            # Sampling, not argmax. A policy gradient method learns a
            # STOCHASTIC policy, and the entropy it retains is part of what it
            # learned; serving only the argmax throws that away and is a
            # different agent from the one you evaluated. /act defaults to
            # deterministic anyway, so this is an opt-in the caller makes.
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
        # Which artifact `policy_name="default"` should mean. From this topic on
        # there is more than one `.npz` in policies/ — the base template's
        # smoke-test table plus the network you actually trained — and the base
        # rule ("default = the only one, if there is exactly one") stops
        # resolving. /rollout and the Streamlit app both send "default", so
        # without this every Play-tab click would 404 the moment a second
        # artifact landed in the directory.
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

        if "default" not in self._cache:
            chosen = None
            if self.default_name and self.default_name in self._cache:
                chosen = self._cache[self.default_name]
            elif len(self._cache) == 1:
                # A fresh clone with one policy still needs no configuration.
                chosen = next(iter(self._cache.values()))
            # Note what does NOT happen here: with several artifacts and no
            # DEFAULT_POLICY set, "default" stays unresolved and /act answers
            # 404 listing the names that exist. Guessing — "the newest one",
            # "the biggest one" — would let a stale artifact serve traffic
            # silently, which is a worse failure than a 404 with a fix in it.
            if chosen is not None:
                self._cache["default"] = chosen

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
            pol = TabularPolicy(Q=np.asarray(z["Q"], dtype=np.float64))
            meta |= {"kind": "tabular", "obs_dim": 1, "n_actions": pol.n_actions}
            return pol, meta

        layers = layers_from_npz(z)
        if not layers:
            raise ValueError(
                f"{path.name}: no 'Q' and no 'W0'/'b0' — see train/export.py for the layout"
            )
        # Checked at LOAD time, not at first request. A transposed matrix or a
        # short bias is a build error, and a build error should surface when the
        # process starts and /healthz reports the artifact unloadable — not
        # halfway through a stakeholder demo.
        check_layer_shapes(layers)
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
