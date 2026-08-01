"""
api/forward.py — the forward pass of a trained neural policy, in NumPy alone.

Read the imports. There is one, and it is NumPy. Three trained agents are served
from this file: two discrete (A2C on CartPole, PPO on Acrobot) and one
continuous (SAC on Pendulum). The arithmetic is genuinely this small:

    discrete            h = relu(W0 @ x + b0); ...; p = softmax(W_last @ h + b)
    squashed Gaussian   same trunk, then split the output into (mean, log_std),
                        squash with tanh, and rescale into the action range

`api/policy.py` wraps these with artifact loading, checksums and the `act()`
contract the service calls. They live in their own module so that
`tests/test_equivalence.py` can compare THEM — not a wrapper around them —
against the PyTorch modules the weights came from.

Weight layout, which is the whole contract with train/export.py:

    W{i} has shape (out_features, in_features)   — PyTorch's nn.Linear layout,
    b{i} has shape (out_features,)                 exported WITHOUT transposing

If an equivalence test fails, that convention is where to look first. The
diagnostic recipe is at the bottom of this file.
"""

from __future__ import annotations

import numpy as np

Layers = list[tuple[np.ndarray, np.ndarray]]

# The bounds SAC clamps its log standard deviation to during training. They must
# be reproduced here byte for byte, because a clamp applied on one side only is
# a difference between the two implementations that the equivalence test will
# report as a mysterious 1e-1 disagreement in the tail of the input space. The
# values travel inside the artifact as well (`log_std_min` / `log_std_max`);
# these are only the fallback for an archive written before that key existed.
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


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
    """Evaluate the ReLU stack and return the OUTPUT LAYER'S RAW VALUES.

    No activation on the last layer, and no softmax here. Three reasons:

      * The SAC actor's output is half mean and half log-standard-deviation; a
        softmax over it would be meaningless.
      * A critic exported with the same helper produces a scalar value estimate,
        not a probability.
      * Keeping logits and probabilities separate gives a later addition — a
        temperature, an action mask — one obvious place to go.

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
    """π(·|s) for a discrete policy — the quantity two of the equivalence tests compare.

    They compare the DISTRIBUTION, not the argmax action. Comparing argmaxes
    would pass for a badly wrong network whenever the wrongness happened not to
    flip the winner, which on a two-action environment is most of the time.
    """
    return softmax(mlp_forward(layers, x), axis=-1)


# ---------------------------------------------------------------------------
# The continuous head. This is the "three extra lines" the product brief means,
# and they are the three below marked (1), (2), (3).
# ---------------------------------------------------------------------------


def gaussian_head(
    layers: Layers,
    x: np.ndarray,
    log_std_min: float = LOG_STD_MIN,
    log_std_max: float = LOG_STD_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the actor's output into a pre-squash mean and a clamped log σ.

    SAC's actor emits `2 * action_dim` numbers: the first half is the mean of a
    Gaussian in PRE-SQUASH space, the second half its log standard deviation.
    The standard deviation is state-dependent — that is the whole point of a
    maximum-entropy actor, which must be able to be uncertain in some states and
    confident in others — so it comes out of the network rather than sitting in
    a separate parameter vector.

    The clamp is load-bearing, not defensive. Unclamped, `exp(log_std)` for a
    log σ of −40 underflows to zero and the log-probability term in the SAC
    objective goes to +inf; for a log σ of 30 it overflows. PyTorch-side SAC
    clamps for exactly this reason, and if the serving path does not apply the
    identical clamp the two implementations compute different functions wherever
    the raw head leaves the interval.
    """
    out = mlp_forward(layers, x)
    half = out.shape[-1] // 2
    if half * 2 != out.shape[-1]:
        raise ValueError(
            f"a squashed-Gaussian actor must emit an EVEN number of outputs "
            f"(mean and log σ), got {out.shape[-1]}. Check the final layer width "
            "in train/nets.SquashedGaussianActor — it is 2 * action_dim."
        )
    mean = out[..., :half]                                              # (1)
    log_std = np.clip(out[..., half:], log_std_min, log_std_max)        # (2)
    return mean, log_std


def squashed_action(
    layers: Layers,
    x: np.ndarray,
    deterministic: bool = True,
    action_scale: np.ndarray | float = 1.0,
    action_bias: np.ndarray | float = 0.0,
    log_std_min: float = LOG_STD_MIN,
    log_std_max: float = LOG_STD_MAX,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """The deployed continuous action: `scale * tanh(mean [+ σ·ε]) + bias`.

    Two things here are worth more than the three lines they cost.

    **`deterministic=True` returns `tanh(mean)`, which is the MODE of the
    squashed distribution and NOT its mean.** tanh is nonlinear, so
    E[tanh(z)] ≠ tanh(E[z]), and the mean of a tanh-squashed Gaussian has no
    closed form at all. Serving the mode is the standard choice and it is the
    right one, but it is a choice: say in your model card that the deployed
    deterministic action is the mode, or a reader will assume you are serving an
    expectation you never computed.

    **The rescaling is not cosmetic.** tanh gives [−1, 1]; Pendulum's torque
    lives in [−2, 2]. An agent deployed without the scale can apply at most half
    the torque it trained with, and the symptom — a policy that swings the
    pendulum most of the way up and then stalls — looks like undertraining
    rather than like a units bug. The scale and bias travel inside the `.npz`
    so the serving path cannot forget them.
    """
    mean, log_std = gaussian_head(layers, x, log_std_min, log_std_max)
    z = mean
    if not deterministic:
        gen = rng or np.random.default_rng()
        # Reparameterised sampling, the same draw SAC trains with: z = μ + σ·ε.
        z = mean + np.exp(log_std) * gen.standard_normal(mean.shape)
    return np.asarray(action_scale) * np.tanh(z) + np.asarray(action_bias)   # (3)


# ---------------------------------------------------------------------------
# Archive plumbing
# ---------------------------------------------------------------------------


def layers_from_npz(z, prefix: str = "") -> Layers:
    """Pull `W0/b0, W1/b1, ...` out of a loaded `.npz` in index order.

    `prefix` exists so one archive can hold more than one network — an actor
    under `W{i}` and a critic under `vW{i}`, say. This product exports only the
    actor, because a critic is a training artifact and the service never calls
    it; keeping the seam here means adding one later does not change the loader.

    Stopping at the first missing index rather than sorting the key list is
    deliberate: a gap (W0, W1, W3) is a broken export, and silently skipping the
    hole would deploy a network with a layer missing and no error anywhere.
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
        (the good case) or — when the hidden width happens to equal obs_dim —
        succeeds and computes a different function forever (the bad case).
      * A bias of the wrong length, which broadcasts instead of raising.

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


def scalar_from_npz(z, key: str, default: float) -> float:
    """Read a 0-d or 1-element float array out of an archive, with a fallback.

    Present so that an artifact exported before a key existed still loads. The
    alternative — raising — would make every previously trained policy in your
    repository unloadable the moment you add a field, which is how a student
    ends up deleting the artifact that produced the numbers in their report.
    """
    if key not in z.files:
        return float(default)
    return float(np.asarray(z[key]).reshape(-1)[0])


def text_from_npz(z, key: str, default: str = "") -> str:
    """Read a string written by `np.savez` as a unicode array.

    Strings are stored as 0-d `<U` arrays, which `np.load(allow_pickle=False)`
    reads happily — worth knowing, because the obvious way to attach metadata
    (a pickled dict) is exactly what `allow_pickle=False` exists to forbid. A
    pickle is arbitrary code execution wearing a data extension, and the
    deployed service must never load one.
    """
    if key not in z.files:
        return default
    return str(np.asarray(z[key]).item()) if np.asarray(z[key]).ndim == 0 else str(z[key])


# ---------------------------------------------------------------------------
# If an equivalence test fails, work this list top to bottom. In practice it is
# the first item nine times out of ten.
#
#   1. Transposed weights. Print `W0.shape` and compare it with
#      `actor.net[0].weight.shape` in PyTorch. They must be IDENTICAL — not
#      transposes of each other.
#
#   2. A missing bias. `export_torch_mlp` writes `b{i}` for every Linear it
#      finds; if your module builds a layer with `bias=False`, the key is absent
#      and `layers_from_npz` stops early, quietly deploying a shorter network.
#      Compare `len(layers)` against the number of Linear layers in the module.
#
#   3. An activation on the output layer. `mlp_forward` deliberately applies
#      none. If your PyTorch module ends in `nn.ReLU()`, `nn.Softmax()` or
#      `nn.Tanh()`, the two implementations compute different functions and both
#      are "right" — fix the module, not this file.
#
#   4. SAC ONLY — the log σ clamp, applied on one side only, or applied with
#      different bounds. The clamp is part of the function, not a safety net.
#
#   5. SAC ONLY — the action scale. If NumPy returns exactly half of what
#      PyTorch returns on Pendulum, the `action_scale=2.0` never made it into
#      the archive.
#
#   6. Observation normalisation applied on one side only. If you exported
#      `obs_mean`/`obs_std`, the serving path must apply them (api/policy.py
#      does) and the PyTorch reference must apply them too.
#
#   7. Only then suspect float precision. float32 weights evaluated in float64
#      differ from float32 end-to-end by roughly 1e-7 on these network sizes. A
#      difference of 1e-3 is a bug, not a rounding error.
# ---------------------------------------------------------------------------
