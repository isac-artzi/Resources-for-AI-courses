"""
train/export.py — turn a trained policy into the artifact the service deploys.

This is the seam between the two tiers, and it is worth stating plainly what
crosses it: an `.npz` archive of float arrays plus a few short strings, and
nothing else. No pickled objects (a pickle is arbitrary code execution and
cannot be reviewed), no framework checkpoint, no optimiser state. The deployed
application must be able to load this file with
`numpy.load(..., allow_pickle=False)`.

Naming convention inside the archive — api/policy.py depends on it:

    tabular  : Q
    network  : W0, b0, W1, b1, ...   (PyTorch nn.Linear weight is (out, in);
                                      export it WITHOUT transposing)
    metadata : head        "categorical" | "squashed_gaussian"
               env_id      "CartPole-v1" | "Acrobot-v1" | "Pendulum-v1"
    continuous only:
               action_scale, action_bias, log_std_min, log_std_max
    optional : obs_mean, obs_std

Note `float32` for the weights. It halves the artifact for no measurable loss of
policy quality, and artifact size is a number you are accountable for.

Note also that the metadata strings are written as NumPy unicode arrays rather
than as a JSON blob or a pickled dict. `np.savez` stores a `<U11` array
perfectly well and `allow_pickle=False` reads it back; a pickled dict would be
refused by exactly the flag that keeps the deployed service from executing
whatever is in the file.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Any

import numpy as np

# Written into every archive and read back by api/policy.py. Kept as module
# constants so that adding a fourth head is a change in two files rather than a
# grep for string literals.
HEAD_CATEGORICAL = "categorical"
HEAD_SQUASHED_GAUSSIAN = "squashed_gaussian"

# Keys that must NOT be cast to float32, because they are text. Listed
# explicitly rather than detected by dtype so that a student who adds a metadata
# field and forgets this line gets a loud failure (a string cast to float32
# raises) rather than a silent one.
TEXT_KEYS = frozenset({"head", "env_id", "algorithm"})


def sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def export_arrays(arrays: dict[str, Any], path: str | pathlib.Path) -> dict[str, Any]:
    """Write a policy archive and return its registry row.

    Everything numeric is stored as float32; everything in `TEXT_KEYS` is stored
    as a unicode array. That split is the whole of the "metadata" mechanism, and
    it is deliberately this small — a schema for artifact metadata is a thing
    you can spend a week on and this product does not need one.
    """
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for k, v in arrays.items():
        payload[k] = np.asarray(v) if k in TEXT_KEYS else np.asarray(v, dtype=np.float32)
    np.savez_compressed(p, **payload)
    return {
        "name": p.stem,
        "format": "npz",
        "bytes": p.stat().st_size,
        "sha256": sha256_of(p),
    }


def export_qtable(
    Q: np.ndarray, path: str | pathlib.Path = "policies/q_table.npz"
) -> dict[str, Any]:
    row = export_arrays({"Q": Q}, path)
    return row | {"kind": "tabular", "obs_dim": 1, "n_actions": int(Q.shape[1])}


def _linear_arrays(module) -> tuple[dict[str, np.ndarray], int]:
    """Walk a torch module and pull out its Linear layers in order.

    Imported inside the function rather than at module scope precisely so that
    this file can be read — and `export_arrays` used — inside the serving
    environment, where torch is not installed.
    """
    import torch  # noqa: F401  (training tier only)

    arrays: dict[str, np.ndarray] = {}
    i = 0
    for layer in module.modules():
        if layer.__class__.__name__ == "Linear":
            arrays[f"W{i}"] = layer.weight.detach().cpu().numpy()
            if layer.bias is None:
                raise ValueError(
                    f"Linear layer {i} was built with bias=False. The archive layout "
                    "pairs W{i} with b{i}, and api/forward.layers_from_npz stops at "
                    "the first missing index — so a bias-free layer would silently "
                    "deploy a SHORTER network. Keep the biases."
                )
            arrays[f"b{i}"] = layer.bias.detach().cpu().numpy()
            i += 1
    if not arrays:
        raise ValueError("no Linear layers found — is this the module you meant to export?")
    return arrays, i


def export_torch_mlp(
    module, path: str | pathlib.Path, env_id: str = "", **extra
) -> dict[str, Any]:
    """Export a categorical actor: a torch.nn.Sequential of Linear/ReLU layers.

    Used for A2C and PPO. The module must emit LOGITS — no softmax on the output
    layer. `api/forward.action_probabilities` applies the softmax, and
    `torch.distributions.Categorical(logits=...)` applies it on the training
    side; a softmax inside the module would make PyTorch apply it twice and
    NumPy once, and the equivalence test would fail for a reason that has
    nothing to do with the export.
    """
    arrays, n = _linear_arrays(module)
    arrays |= {"head": HEAD_CATEGORICAL, "env_id": env_id}
    arrays |= {k: np.asarray(v) for k, v in extra.items()}
    row = export_arrays(arrays, path)
    return row | {
        "kind": "mlp",
        "obs_dim": int(arrays["W0"].shape[1]),
        "n_actions": int(arrays[f"W{n-1}"].shape[0]),
        "action_space": "discrete",
        "env_id": env_id or None,
    }


def export_squashed_gaussian(
    module,
    path: str | pathlib.Path,
    env_id: str,
    action_scale,
    action_bias=0.0,
    log_std_min: float = -20.0,
    log_std_max: float = 2.0,
    **extra,
) -> dict[str, Any]:
    """Export the SAC actor: the same trunk, a `2 * action_dim` output, a tanh squash.

    Four things travel with the weights, and each of them is a bug that has
    happened to somebody:

      * `head` — so the loader knows the output is (mean, log σ) and not a pair
        of logits. Without it a two-output Pendulum actor is indistinguishable
        from a two-action discrete policy, and `argmax` over a mean and a log
        standard deviation is a perfectly well-typed way to be completely wrong.
      * `action_scale` / `action_bias` — tanh gives [−1, 1]; Pendulum's torque is
        [−2, 2]. A policy deployed without the scale applies half the torque it
        trained with and looks undertrained rather than broken.
      * `log_std_min` / `log_std_max` — the clamp is part of the function the
        network computes, not a safety net around it. Applied on one side only,
        the two implementations disagree exactly where the raw head is extreme,
        which is the tail of the input space where you will not look.

    The critic is deliberately NOT exported. The service never calls it: a
    deployed actor chooses actions, and the twin Q-networks exist to train it.
    Shipping them would triple the artifact and put a training-time object into
    a production bundle.
    """
    arrays, n = _linear_arrays(module)
    out_dim = int(arrays[f"W{n-1}"].shape[0])
    if out_dim % 2 != 0:
        raise ValueError(
            f"a squashed-Gaussian actor must emit 2 * action_dim outputs; this one "
            f"emits {out_dim}, which is odd. Check the final layer width."
        )
    arrays |= {
        "head": HEAD_SQUASHED_GAUSSIAN,
        "env_id": env_id,
        "action_scale": np.atleast_1d(np.asarray(action_scale, dtype=np.float32)),
        "action_bias": np.atleast_1d(np.asarray(action_bias, dtype=np.float32)),
        "log_std_min": np.asarray([log_std_min], dtype=np.float32),
        "log_std_max": np.asarray([log_std_max], dtype=np.float32),
    }
    arrays |= {k: np.asarray(v) for k, v in extra.items()}
    row = export_arrays(arrays, path)
    return row | {
        "kind": "mlp",
        "obs_dim": int(arrays["W0"].shape[1]),
        "n_actions": out_dim // 2,
        "action_space": "continuous",
        "env_id": env_id,
    }


def register(row: dict[str, Any], experiment_id: str | None = None) -> None:
    """Write the artifact row to the `policies` table.

    Called AFTER the training run has been written, so `policies.experiment_id`
    points at a row that exists and "which run produced the thing we deployed"
    stays answerable six weeks later.

    Note the filter: `policies` in 001_init.sql has no `action_space` or
    `env_id` column of its own in the base schema, and 002_topic4.sql adds them.
    Sending a key the table does not have is a 400 from PostgREST, so the
    columns the row may carry are listed rather than assumed.
    """
    from shared.store import get_store

    allowed = {
        "name", "format", "bytes", "sha256", "kind", "obs_dim", "n_actions",
        "action_space", "env_id",
    }
    payload = {k: v for k, v in row.items() if k in allowed}
    get_store().insert_policy(payload | {"experiment_id": experiment_id})
