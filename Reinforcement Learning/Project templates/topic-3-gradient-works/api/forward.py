"""
api/forward.py — the forward pass of a trained neural policy, in NumPy alone.

This is the file the whole no-torch rule is built around, and Topic 3 is the
first topic where it does real work: the deployed policy stopped being a table
and became a two-hidden-layer network. Read the imports. There is one, and it
is NumPy.

The arithmetic is genuinely this small:

    h = relu(W0 @ x + b0)
    h = relu(W1 @ h + b1)
    p = softmax(W2 @ h + b2)

`api/policy.py` wraps these functions with artifact loading, checksums and the
`act()` contract the service tier calls. They live in their own module so that
`tests/test_equivalence.py` can compare THEM — not a wrapper around them —
against the PyTorch module the weights came from.

Weight layout, which is the whole contract with train/export.py:

    W{i} has shape (out_features, in_features)   — PyTorch's nn.Linear layout,
    b{i} has shape (out_features,)                 exported WITHOUT transposing

If the equivalence test fails, that convention is where to look first. See the
diagnostic recipe at the bottom of this file.
"""

from __future__ import annotations

import numpy as np

Layers = list[tuple[np.ndarray, np.ndarray]]


def relu(x: np.ndarray) -> np.ndarray:
    """max(0, x), elementwise.

    Written as `np.maximum(0.0, x)` rather than `x * (x > 0)` because the second
    form silently promotes an integer input to a boolean product and returns a
    different dtype than it was given.
    """
    return np.maximum(0.0, x)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax over `axis`.

    Subtracting the max is not a nicety. Without it, a logit of 800 — which a
    diverging policy will produce, and which you will only meet in production —
    overflows to `inf`, the whole vector becomes NaN, and the service returns
    500 on an input the training run never saw.

    `axis=-1` with `keepdims=True` makes this work unchanged on a single
    observation of shape (n_actions,) and on a batch of shape (N, n_actions).
    The equivalence test needs the batched form; the service only ever uses the
    single-observation form. One implementation for both means the thing you
    tested is the thing you deployed.
    """
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def mlp_forward(layers: Layers, x: np.ndarray) -> np.ndarray:
    """Evaluate the ReLU stack and return the OUTPUT LAYER'S RAW VALUES (logits).

    No activation on the last layer, and no softmax here. Two reasons:

      * The value network exported by train/baseline.py shares this code and its
        output is a scalar value estimate, not a probability.
      * Keeping logits and probabilities separate means a later addition — a
        temperature, an action mask — has one obvious place to go.

    `x` may be a single observation of shape (obs_dim,) or a batch of shape
    (N, obs_dim). The `x @ W.T` form handles both, which is why it is written
    that way rather than as the `W @ x` you would write on a whiteboard.
    """
    h = np.asarray(x, dtype=np.float64)
    last = len(layers) - 1
    for i, (W, b) in enumerate(layers):
        h = h @ np.asarray(W, dtype=np.float64).T + np.asarray(b, dtype=np.float64)
        if i < last:
            h = relu(h)
    return h


def action_probabilities(layers: Layers, x: np.ndarray) -> np.ndarray:
    """π(·|s) for a discrete policy — the quantity the equivalence test compares.

    The test compares the DISTRIBUTION, not the argmax action. Comparing
    argmaxes would pass for a badly wrong network whenever the wrongness
    happened not to flip the winner, which on a two-action environment is most
    of the time.
    """
    return softmax(mlp_forward(layers, x), axis=-1)


def layers_from_npz(z, prefix: str = "") -> Layers:
    """Pull `W0/b0, W1/b1, ...` out of a loaded `.npz` in index order.

    `prefix` exists so one archive can hold both the policy network and the
    value network (`vW0`, `vb0`, ...) — Topic 4 ships an actor and a critic in
    a single artifact and this is the seam it uses.

    Stopping at the first missing index rather than sorting the key list is
    deliberate: a gap (W0, W1, W3) is a broken export, and silently skipping
    the hole would deploy a network with a layer missing.
    """
    layers: Layers = []
    i = 0
    while f"{prefix}W{i}" in z.files and f"{prefix}b{i}" in z.files:
        layers.append(
            (
                np.asarray(z[f"{prefix}W{i}"], dtype=np.float64),
                np.asarray(z[f"{prefix}b{i}"], dtype=np.float64),
            )
        )
        i += 1
    return layers


def check_layer_shapes(layers: Layers) -> None:
    """Fail loudly at LOAD time on the two mistakes that produce silent nonsense.

    Both are worth catching here rather than at the first request:

      * A transposed weight matrix. `W0` is (out, in). If you exported
        `layer.weight.T`, the very first matmul either raises a shape error
        (the good case) or, when hidden width happens to equal obs_dim,
        succeeds and computes a different function forever (the bad case).
      * A bias of the wrong length, which broadcasts instead of adding.

    NumPy's broadcasting rules are what make the bad cases silent, so the check
    is explicit rather than left to the arithmetic.
    """
    for i, (W, b) in enumerate(layers):
        if W.ndim != 2:
            raise ValueError(f"W{i} has shape {W.shape}; expected a 2-D (out, in) matrix")
        if b.shape != (W.shape[0],):
            raise ValueError(
                f"b{i} has shape {b.shape} but W{i} produces {W.shape[0]} outputs. "
                "A bias whose length does not match the layer width will broadcast "
                "rather than raise, so check this rather than trusting the matmul."
            )
        if i > 0 and W.shape[1] != layers[i - 1][0].shape[0]:
            raise ValueError(
                f"W{i} expects {W.shape[1]} inputs but W{i-1} produces "
                f"{layers[i-1][0].shape[0]}. This is the transposed-export bug: "
                "PyTorch stores nn.Linear.weight as (out, in) and export.py must "
                "write it out without transposing."
            )


# ---------------------------------------------------------------------------
# If tests/test_equivalence.py fails, work this list top to bottom. In practice
# it is the first item nine times out of ten.
#
#   1. Transposed weights. Print `W0.shape` and compare it with
#      `policy.net[0].weight.shape` in PyTorch. They must be IDENTICAL — not
#      transposes of each other.
#
#   2. A missing bias. `export_torch_mlp` writes `b{i}` for every Linear it
#      finds; if your module builds a layer with `bias=False`, the key is absent
#      and `layers_from_npz` stops early, quietly deploying a shorter network.
#      Compare `len(layers)` with the number of Linear layers in the module.
#
#   3. An activation on the output layer. `mlp_forward` deliberately applies
#      none. If your PyTorch module ends in `nn.ReLU()` or `nn.Softmax()`, the
#      two implementations are computing different functions and both are
#      "right" — fix the module, not this file.
#
#   4. Observation normalisation applied on one side only. If you exported
#      `obs_mean`/`obs_std`, the serving path must apply them (api/policy.py
#      does) and the PyTorch reference must apply them too.
#
#   5. Only then suspect float precision. float32 weights evaluated in float64
#      differ from float32 end-to-end by roughly 1e-7 on this network size. A
#      difference of 1e-3 is a bug, not a rounding error.
# ---------------------------------------------------------------------------
