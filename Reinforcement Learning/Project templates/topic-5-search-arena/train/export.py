"""
train/export.py — turn a trained policy into the artifact the service deploys.

This is the seam between the two tiers, and it is worth stating plainly what
crosses it: an `.npz` archive of float arrays, and nothing else. No pickled
objects (a pickle is arbitrary code execution and cannot be reviewed), no
framework checkpoint, no optimiser state. The deployed application must be
able to load this file with `numpy.load(..., allow_pickle=False)`.

Naming convention inside the archive — api/policy.py depends on it:

    tabular : Q
    network : W0, b0, W1, b1, ...   (PyTorch nn.Linear weight is (out, in);
                                     export it WITHOUT transposing)
    optional: log_std, obs_mean, obs_std

Note `float32`. It halves the artifact for no measurable loss of policy
quality, and artifact size is a number you are accountable for.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Any

import numpy as np


def sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def export_arrays(arrays: dict[str, np.ndarray], path: str | pathlib.Path) -> dict[str, Any]:
    """Write a policy archive and return its registry row."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **{k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()})
    return {
        "name": p.stem,
        "format": "npz",
        "bytes": p.stat().st_size,
        "sha256": sha256_of(p),
    }


def export_qtable(Q: np.ndarray, path: str | pathlib.Path = "policies/q_table.npz") -> dict[str, Any]:
    row = export_arrays({"Q": Q}, path)
    return row | {"kind": "tabular", "obs_dim": 1, "n_actions": int(Q.shape[1])}


def export_torch_mlp(module, path: str | pathlib.Path, **extra: np.ndarray) -> dict[str, Any]:
    """Export a torch.nn.Sequential of Linear/ReLU layers.

    Imported here rather than at module scope precisely so that this file can
    be read — and its other functions used — inside the serving environment,
    where torch is not installed.
    """
    import torch  # noqa: F401  (training tier only)

    arrays: dict[str, np.ndarray] = {}
    i = 0
    for layer in module.modules():
        if layer.__class__.__name__ == "Linear":
            arrays[f"W{i}"] = layer.weight.detach().cpu().numpy()
            arrays[f"b{i}"] = layer.bias.detach().cpu().numpy()
            i += 1
    if not arrays:
        raise ValueError("no Linear layers found — is this the module you meant to export?")
    arrays |= {k: np.asarray(v) for k, v in extra.items()}
    row = export_arrays(arrays, path)
    W_first = arrays["W0"]
    W_last = arrays[f"W{i-1}"]
    return row | {"kind": "mlp", "obs_dim": int(W_first.shape[1]), "n_actions": int(W_last.shape[0])}


def register(row: dict[str, Any], experiment_id: str | None = None) -> None:
    """Write the artifact row to the `policies` table."""
    from shared.store import get_store

    get_store().insert_policy(row | {"experiment_id": experiment_id})


def export_policy_value_net(
    arrays: dict[str, np.ndarray], path: str | pathlib.Path = "policies/alphazero_c4.npz"
) -> dict[str, Any]:
    """Export the Topic 5 policy-value network.

    The archive layout is a contract with `search/net.py` and with
    `api/policy.py`'s loader, so it is validated HERE rather than discovered at
    load time in the service. An archive that is missing a head is a broken
    deployment; an export that refuses to write one is a failed build, and a
    failed build is enormously cheaper.

    Note this writes float32 like every other export in this course. The
    accuracy cost is unmeasurable — the value head's output is squashed to
    [-1, 1] and the policy head is fed to a softmax — and the artifact halves.
    Artifact size is a number you are accountable for.
    """
    required = ("W0", "b0", "W1", "b1", "Wp", "bp", "Wv", "bv")
    missing = [k for k in required if k not in arrays]
    if missing:
        raise ValueError(
            f"policy-value export is missing {missing}. Expected keys: {required}. "
            "See train/selfplay.py::_extract_arrays."
        )
    row = export_arrays({k: arrays[k] for k in required}, path)
    return row | {
        "kind": "value-net",
        "obs_dim": int(np.asarray(arrays["W0"]).shape[1]),
        "n_actions": int(np.asarray(arrays["Wp"]).shape[0]),
    }
