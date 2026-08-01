"""
api/policy.py — artifact discovery, and the `act()` contract the service calls.

Read the imports. There are three, and none of them is a machine-learning
framework. This is the file that makes the no-torch rule cost nothing: a trained
policy in this course is either a lookup table or three matrix multiplies, and
both are a handful of lines of NumPy.

What lives here:
  * `PolicyArtifactStore` — discovers `.npz` archives in POLICY_DIR, checksums
    them, and keeps them loaded.
  * `TabularPolicy`  — Topics 1, 2 and 5: argmax over a row of a Q-table.
  * `MLPPolicy`      — Topics 3, 4 and 6: a ReLU stack with either a softmax
                       head (A2C, PPO) or a tanh-squashed Gaussian head (SAC).

The arithmetic itself lives in `api/forward.py`. It is the same arithmetic; it
is in its own module so the required NumPy/PyTorch equivalence tests can
exercise the exact functions the service calls rather than a copy that could
drift.

What does NOT live here: any training loop, any optimiser, any gradient. If you
find yourself importing something to fit a parameter, you are in the wrong
directory — that belongs in train/.

Topic 4 adds ONE field to the artifact metadata and it is the field the whole
product turns on: **`env_id`**. Three agents now sit behind one `/act`, their
observations are 4, 6 and 3 numbers wide, and the service has to know which is
which without constructing a Gymnasium environment. The id is written into the
archive at export time and read back here, so `/rollout` can build the right
environment and `/act` can reject the wrong observation with a 422 that names
both numbers.
"""

from __future__ import annotations

import hashlib
import pathlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from api.forward import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    action_probabilities,
    check_layer_shapes,
    gaussian_head,
    layers_from_npz,
    mlp_forward,
    relu,
    scalar_from_npz,
    softmax,
    squashed_action,
    text_from_npz,
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
    """Feed-forward policy exported from PyTorch, with one of two heads.

    Expected archive keys: `W0, b0, W1, b1, ...` in order, plus

        head          "categorical" (A2C, PPO) or "squashed_gaussian" (SAC)
        env_id        which environment this policy was trained on
        action_scale  continuous only: tanh gives [-1, 1]; the action is not
        action_bias   continuous only
        log_std_min   continuous only: the clamp SAC trained under
        log_std_max
        obs_mean      optional observation normalisation, computed at TRAINING
        obs_std       time and shipped alongside so it cannot drift

    The weight convention is PyTorch's own: `nn.Linear.weight` has shape
    (out, in), and `train/export.py` writes it out WITHOUT transposing. If an
    equivalence test fails, that convention is the first thing to check — a
    transposed matrix or a dropped bias accounts for nearly every failure.
    """

    layers: list[tuple[np.ndarray, np.ndarray]]
    head: str = "categorical"
    env_id: str = ""
    action_scale: np.ndarray = field(default_factory=lambda: np.asarray(1.0))
    action_bias: np.ndarray = field(default_factory=lambda: np.asarray(0.0))
    log_std_min: float = LOG_STD_MIN
    log_std_max: float = LOG_STD_MAX
    obs_mean: np.ndarray | None = None
    obs_std: np.ndarray | None = None

    kind = "mlp"

    @property
    def discrete(self) -> bool:
        return self.head == "categorical"

    @property
    def obs_dim(self) -> int:
        return int(self.layers[0][0].shape[1])

    @property
    def n_actions(self) -> int:
        """Number of discrete actions, or the width of the continuous action vector.

        Halved for the Gaussian head, because that head emits a mean AND a log
        standard deviation for every action dimension. Reporting the raw output
        width would tell `/policies` — and therefore your model card — that the
        Pendulum agent has two actions when it has one.
        """
        out = int(self.layers[-1][0].shape[0])
        return out if self.discrete else out // 2

    # -- the forward pass, in the shapes the service needs ------------------

    def _normalise(self, x: np.ndarray) -> np.ndarray:
        """Apply the observation statistics computed at TRAINING time.

        They travel inside the archive precisely so this cannot drift.
        Recomputing them from live traffic is the classic train/serve skew bug —
        see the note in shared/preprocess.py.
        """
        x = np.asarray(x, dtype=np.float64)
        if self.obs_mean is not None and self.obs_std is not None:
            x = (x - self.obs_mean) / (self.obs_std + 1e-8)
        return x

    def probabilities(self, state: np.ndarray) -> np.ndarray:
        """π(·|s) as a vector. Two of the three equivalence tests compare this.

        Exposed as its own method because a *distribution* is what you can
        compare against PyTorch to a tolerance. Comparing sampled actions
        compares two random number generators, and comparing argmaxes passes for
        a badly wrong network whenever the error does not happen to flip the
        winner — which, with two or three actions, is most of the time.
        """
        if not self.discrete:
            raise ValueError(
                "probabilities() is only defined for a categorical policy; this "
                f"artifact has a {self.head!r} head. Compare mean and log σ "
                "instead — see gaussian_parameters()."
            )
        return action_probabilities(self.layers, self._normalise(state))

    def gaussian_parameters(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(pre-squash mean, clamped log σ). The SAC equivalence test compares this.

        Compared BEFORE the squash as well as after, deliberately. tanh is a
        contraction: it maps a mean of 8.0 and a mean of 12.0 to 0.99999977 and
        0.99999999, so a serious disagreement in the pre-squash mean shows up
        after the squash as a difference of 2e-7 and would pass any sane
        tolerance. Checking the raw head is what makes the SAC equivalence test
        as strong as the two discrete ones.
        """
        if self.discrete:
            raise ValueError("gaussian_parameters() is only defined for a continuous policy")
        return gaussian_head(
            self.layers, self._normalise(state), self.log_std_min, self.log_std_max
        )

    def act(self, state: np.ndarray, deterministic: bool = True):
        x = np.asarray(state, dtype=np.float64)
        if x.ndim != 1 or x.shape[0] != self.obs_dim:
            raise ValueError(
                f"observation has dimension {x.shape[0]}, artifact expects {self.obs_dim}"
            )
        xn = self._normalise(x)

        if self.discrete:
            p = softmax(mlp_forward(self.layers, xn))
            if deterministic:
                return int(np.argmax(p)), float(np.max(p))
            # Sampling, not argmax. An actor-critic method learns a STOCHASTIC
            # policy and the entropy it retained is part of what it learned;
            # serving only the argmax throws that away and is a different agent
            # from the one you evaluated. `/act` defaults to deterministic, so
            # this is an opt-in the caller makes explicitly.
            return int(np.random.choice(len(p), p=p)), float(np.max(p))

        a = squashed_action(
            self.layers,
            xn,
            deterministic=deterministic,
            action_scale=self.action_scale,
            action_bias=self.action_bias,
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
        )
        # The "value estimate" reported for a continuous policy is the mean
        # log σ, not a Q-value: the deployed artifact is the ACTOR alone, and
        # the critic stayed in the training tier where it belongs. Mean log σ is
        # the honest thing to surface — it is how uncertain the policy is in
        # this state, which is exactly what a maximum-entropy method trades off.
        _, log_std = self.gaussian_parameters(x)
        return [float(v) for v in np.atleast_1d(a)], float(np.mean(log_std))


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

    The checksum is not decoration. `/act` returns it and `audit_log` stores it,
    so six weeks later you can answer "which artifact produced this action" with
    a join rather than with a guess.
    """

    def __init__(
        self,
        policy_dir: str | pathlib.Path = "policies",
        default_name: str | None = None,
    ) -> None:
        self.dir = pathlib.Path(policy_dir)
        # Which artifact `policy_name="default"` should mean. This product ships
        # FOUR `.npz` files — three agents plus the base template's smoke-test
        # table — so the base rule ("default = the only one, if there is exactly
        # one") stops resolving. `/rollout` and the Streamlit Play tab both send
        # "default", so without this every click would 404.
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
            # DEFAULT_POLICY set, "default" stays unresolved and `/act` answers
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
            meta |= {
                "kind": "tabular",
                "obs_dim": 1,
                "n_actions": pol.n_actions,
                "action_space": "discrete",
                "env_id": text_from_npz(z, "env_id") or None,
            }
            return pol, meta

        layers = layers_from_npz(z)
        if not layers:
            raise ValueError(
                f"{path.name}: no 'Q' and no 'W0'/'b0' — see train/export.py for the layout"
            )
        # Checked at LOAD time, not at first request. A transposed matrix or a
        # short bias is a build error, and a build error should surface when the
        # process starts and `/healthz` reports the artifact unloadable — not
        # halfway through a stakeholder demo.
        check_layer_shapes(layers)

        # `head` is READ from the archive rather than inferred. The base template
        # inferred "continuous" from the presence of a `log_std` key, which
        # worked while the course had one continuous shape in it. SAC's log σ is
        # a network OUTPUT rather than a stored parameter, so there is no such
        # key to test for, and the inference would quietly classify the Pendulum
        # actor as a two-action discrete policy and serve `argmax` of a mean and
        # a log standard deviation. Nothing would raise.
        head = text_from_npz(z, "head", "categorical")
        if head not in ("categorical", "squashed_gaussian"):
            raise ValueError(
                f"{path.name}: unknown head {head!r}. Legal values are "
                "'categorical' and 'squashed_gaussian'; see train/export.py."
            )

        pol = MLPPolicy(
            layers=layers,
            head=head,
            env_id=text_from_npz(z, "env_id"),
            action_scale=(
                np.asarray(z["action_scale"], dtype=np.float64)
                if "action_scale" in keys
                else np.asarray(1.0)
            ),
            action_bias=(
                np.asarray(z["action_bias"], dtype=np.float64)
                if "action_bias" in keys
                else np.asarray(0.0)
            ),
            log_std_min=scalar_from_npz(z, "log_std_min", LOG_STD_MIN),
            log_std_max=scalar_from_npz(z, "log_std_max", LOG_STD_MAX),
            obs_mean=np.asarray(z["obs_mean"], dtype=np.float64) if "obs_mean" in keys else None,
            obs_std=np.asarray(z["obs_std"], dtype=np.float64) if "obs_std" in keys else None,
        )
        meta |= {
            "kind": "mlp",
            "obs_dim": pol.obs_dim,
            "n_actions": pol.n_actions,
            "action_space": "discrete" if pol.discrete else "continuous",
            "env_id": pol.env_id or None,
        }
        return pol, meta

    # -- accessors ----------------------------------------------------------

    def names(self) -> list[str]:
        return [n for n in self._cache if n != "default"]

    def metadata(self) -> list[dict[str, Any]]:
        seen, out = set(), []
        for _name, (_, meta) in self._cache.items():
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
