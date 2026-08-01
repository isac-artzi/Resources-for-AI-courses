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


def export_reward_head(
    module,
    vocab: list[str],
    idf: np.ndarray,
    path: str | pathlib.Path,
) -> dict[str, Any]:
    """Export a TF-IDF reward head — WEIGHTS AND FEATURISER TOGETHER, in one archive.

    This is the only export in the course that ships a preprocessing step
    inside the artifact, and the reason is worth stating: the head's input is
    not a raw observation, it is a vector produced by a vocabulary and an IDF
    table that were FITTED ON THE TRAINING SPLIT. Ship the weights alone and
    the serving tier has to re-derive those from somewhere — from live traffic,
    from a second file, from a scikit-learn pickle — and every one of those
    options is a train/serve skew waiting to happen. One archive, one checksum,
    one thing to deploy.

    `vocab` is written as a NumPy unicode array so that the archive stays
    loadable with `allow_pickle=False`. A list of Python strings would be
    pickled, and a pickle in a serving path is arbitrary code execution wearing
    a model's name.

    Note that this function does NOT cast `vocab` to float32 — hence the manual
    `np.savez_compressed` rather than a call to `export_arrays`, which casts
    everything. A float32 vocabulary is a very short and very confusing debug
    session.
    """
    import torch  # noqa: F401  (training tier only)

    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    i = 0
    for layer in module.modules():
        if layer.__class__.__name__ == "Linear":
            # PyTorch stores nn.Linear.weight as (out_features, in_features).
            # Exported WITHOUT transposing, because api/reward.py evaluates
            # `W @ x + b`. If the equivalence test fails, this line is the
            # first suspect.
            arrays[f"W{i}"] = layer.weight.detach().cpu().numpy().astype(np.float32)
            arrays[f"b{i}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
            i += 1
    if not arrays:
        raise ValueError("no Linear layers found — is this the module you meant to export?")
    if arrays[f"W{i - 1}"].shape[0] != 1:
        raise ValueError(
            f"a reward head must output one scalar; this module's last layer "
            f"outputs {arrays[f'W{i - 1}'].shape[0]}"
        )
    if arrays["W0"].shape[1] != len(vocab):
        raise ValueError(
            f"W0 expects {arrays['W0'].shape[1]} features but the vocabulary has "
            f"{len(vocab)} — the vectoriser and the head disagree"
        )

    np.savez_compressed(
        p,
        vocab=np.asarray(vocab, dtype=np.str_),
        idf=np.asarray(idf, dtype=np.float32),
        **arrays,
    )
    return {
        "name": p.stem,
        "format": "npz",
        "bytes": p.stat().st_size,
        "sha256": sha256_of(p),
        "kind": "reward-head",
        "obs_dim": len(vocab),
        "n_actions": None,
    }


def export_embedding_head(module, dim: int, path: str | pathlib.Path) -> dict[str, Any]:
    """Export the embedding-based reward head. Registered, but NOT deployable.

    It exports to the same `.npz` format as everything else and registers in
    `policies` exactly like every other artifact — the build step requires
    both. What it cannot do is serve, because computing its input needs the
    encoder, and the encoder is 90 MB of weights plus PyTorch.

    Registering an artifact you cannot serve is not a contradiction; it is the
    honest record of a model you trained and chose not to ship. `GET /policies`
    lists it as kind 'mlp' with `obs_dim` equal to the encoder dimension, which
    is enough for a reader to work out why it is not the default.
    """
    row = export_torch_mlp(module, path)
    return row | {"kind": "mlp", "obs_dim": dim, "n_actions": 1}


def register(row: dict[str, Any], experiment_id: str | None = None) -> None:
    """Write the artifact row to the `policies` table."""
    from shared.store import get_store

    get_store().insert_policy(row | {"experiment_id": experiment_id})
