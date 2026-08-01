"""
THE REQUIRED TEST OF THIS TOPIC, three times over: the NumPy forward pass must
reproduce the PyTorch policy, for EACH of the three deployed agents.

Everything else in this repository can be right while this is wrong, and the
symptom is an agent that scored −90 on Acrobot in training and behaves randomly
in the deployed app — with no error anywhere, because a transposed matrix is
still a matrix and NumPy will happily multiply it. That failure has cost more
student weekends than any other in this course, so it is a test rather than a
step in a checklist.

TOLERANCES, and why they are different numbers
----------------------------------------------
    A2C  (CartPole, 4→2 categorical)   1e-5 on max |Δ probability|
    PPO  (Acrobot,  6→3 categorical)   1e-5 on max |Δ probability|
    SAC  (Pendulum, 3→2 Gaussian head) 1e-5 on max |Δ mean|, |Δ log σ|
                                       2e-5 on max |Δ action|

Each number is chosen, not copied. `train/export.py` writes float32 weights and
`api/forward.py` evaluates them in float64, so the two implementations differ by
accumulated rounding of order 1e-7 on these network sizes. 1e-5 is two orders of
magnitude above that noise and several orders below any real bug: a transposed
weight matrix, a missing bias or a doubled softmax all move the output by 1e-2
or more.

The SAC action tolerance is LOOSER than its mean tolerance, and deliberately so:
the action is `2·tanh(mean)`, and the factor of 2 doubles whatever error the
mean carried. A tolerance that did not account for the action scale would be
tighter for the action than for the quantity it is computed from, which is not a
defensible bar.

WHAT EACH TEST COMPARES, AND WHY NOT SOMETHING SIMPLER
-------------------------------------------------------
The categorical tests compare DISTRIBUTIONS, not actions. Comparing argmaxes
would pass for a badly wrong network whenever the error did not happen to flip
the winner, and with two or three actions that is most of the time.

The SAC test compares the PRE-SQUASH mean and log σ, not only the final action.
tanh is a contraction: it maps a mean of 8.0 and a mean of 12.0 to 0.99999977
and 0.99999999, so a catastrophic disagreement in the raw head shows up after
the squash as a difference of 2e-7 and would pass any sane tolerance. Testing
only the action would give you a test that cannot fail.

WHEN ONE FAILS, in order of how often each is the cause — the full diagnostic
recipe is at the bottom of api/forward.py:

    1. A transposed weight matrix.
    2. A missing bias — a layer built with bias=False.
    3. An activation on the output layer of the PyTorch module.
    4. SAC only: the log σ clamp applied on one side, or with other bounds.
    5. SAC only: `action_scale` missing, so NumPy returns exactly half.
    6. Only then, precision.
"""

from __future__ import annotations

import numpy as np
import pytest

from api.forward import (
    action_probabilities,
    check_layer_shapes,
    gaussian_head,
    layers_from_npz,
    scalar_from_npz,
    squashed_action,
    text_from_npz,
)
from api.policy import PolicyArtifactStore

PROB_TOLERANCE = 1e-5
GAUSSIAN_TOLERANCE = 1e-5
ACTION_TOLERANCE = 2e-5


@pytest.fixture(scope="module")
def reference(tmp_path_factory, request):
    """Export three seeded policies and their PyTorch outputs, from a subprocess.

    The subprocess is the point — see `run_torch_script` in conftest.py. This
    test process must remain free of torch or tests/test_no_torch.py becomes a
    lie.

    One subprocess for all three, not three: the interpreter spawn plus the
    torch import is about two seconds, and paying it once for a module-scoped
    fixture keeps the suite fast enough that people actually run it.
    """
    request.getfixturevalue("requires_torch")
    from tests.conftest import run_torch_script

    out = tmp_path_factory.mktemp("equivalence")
    run_torch_script(
        f"""
        from train.nets import dump_reference
        print(json.dumps(dump_reference({str(out)!r}, seed=0, n=256)))
        """
    )
    return out


def _layers(out_dir, stem):
    z = np.load(out_dir / f"equivalence_{stem}.npz", allow_pickle=False)
    layers = layers_from_npz(z)
    check_layer_shapes(layers)
    return z, layers


# ---------------------------------------------------------------------------
# The two categorical agents
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stem", "env_id", "obs_dim", "n_actions"),
    [("cartpole", "CartPole-v1", 4, 2), ("acrobot", "Acrobot-v1", 6, 3)],
)
def test_numpy_reproduces_the_torch_action_distribution(
    reference, stem, env_id, obs_dim, n_actions
):
    z, layers = _layers(reference, stem)
    ref = np.load(reference / f"reference_{stem}.npz")

    assert text_from_npz(z, "env_id") == env_id, (
        "the archive does not record the environment it was trained on; /rollout "
        "cannot construct the right environment without it"
    )
    assert layers[0][0].shape[1] == obs_dim
    assert layers[-1][0].shape[0] == n_actions

    ours = action_probabilities(layers, ref["states"])
    theirs = ref["probs"]
    assert ours.shape == theirs.shape, (
        f"shape mismatch {ours.shape} vs {theirs.shape} — the NumPy forward pass "
        "is not even producing one distribution per observation"
    )

    max_abs_diff = float(np.max(np.abs(ours - theirs)))
    assert max_abs_diff < PROB_TOLERANCE, (
        f"[{env_id}] max |NumPy − PyTorch| = {max_abs_diff:.3e}, tolerance "
        f"{PROB_TOLERANCE:.0e}.\n"
        "This is almost always a TRANSPOSED WEIGHT MATRIX or a MISSING BIAS. Check:\n"
        f"    python -c \"import numpy as np; z=np.load('policies/{stem}.npz'); "
        'print({k: z[k].shape for k in z.files})"\n'
        "and compare each W{i}.shape against the corresponding\n"
        "    actor.net[2*i].weight.shape\n"
        "in PyTorch. They must MATCH, not transpose. See api/forward.py."
    )
    # Printed so the measured number lands in the report rather than only the
    # verdict. "It passed" is weaker evidence than "it agreed to 3e-8".
    print(f"[{env_id}] max |NumPy − PyTorch| = {max_abs_diff:.3e} "
          f"(tolerance {PROB_TOLERANCE:.0e})")


@pytest.mark.parametrize("stem", ["cartpole", "acrobot"])
def test_the_probabilities_are_a_distribution(reference, stem):
    """Rows sum to 1 and nothing is NaN.

    A separate assertion because the difference test can pass while both
    implementations are broken in the same direction — if the exported weights
    were all zeros, both would return a uniform distribution everywhere and
    agree perfectly.
    """
    _z, layers = _layers(reference, stem)
    ref = np.load(reference / f"reference_{stem}.npz")
    ours = action_probabilities(layers, ref["states"])

    assert np.all(np.isfinite(ours))
    assert np.allclose(ours.sum(axis=-1), 1.0, atol=1e-9)
    assert ours.min() >= 0.0
    n = ours.shape[-1]
    assert float(np.abs(ours - 1.0 / n).max()) > 1e-3, (
        f"every probability is 1/{n}, so this test agreed about nothing. Did the "
        "export write zeros?"
    )


# ---------------------------------------------------------------------------
# The SAC agent
# ---------------------------------------------------------------------------


def test_numpy_reproduces_the_torch_squashed_gaussian(reference):
    """The three-line continuous head, checked at every stage.

    Mean, log σ and action are asserted separately rather than as one number.
    Which of the three disagrees tells you where the bug is: a mean mismatch is a
    weight problem, a log σ mismatch with a correct mean is the clamp, and a
    correct mean and log σ with a wrong action is the `action_scale`.
    """
    z, layers = _layers(reference, "pendulum")
    ref = np.load(reference / "reference_pendulum.npz")

    assert text_from_npz(z, "head") == "squashed_gaussian", (
        "without the `head` key the loader cannot tell a 2-output Gaussian actor "
        "from a 2-action discrete policy, and argmax over (mean, log σ) is a "
        "perfectly well-typed way to be completely wrong"
    )
    assert layers[-1][0].shape[0] == 2, "output must be 2 * action_dim = 2 for Pendulum"

    log_std_min = scalar_from_npz(z, "log_std_min", -20.0)
    log_std_max = scalar_from_npz(z, "log_std_max", 2.0)
    scale = np.asarray(z["action_scale"], dtype=np.float64)
    bias = np.asarray(z["action_bias"], dtype=np.float64)
    assert float(scale.reshape(-1)[0]) == pytest.approx(2.0), (
        "Pendulum's torque is in [-2, 2] and tanh gives [-1, 1]. Without "
        "action_scale=2 the deployed agent can apply half the torque it trained "
        "with, and the symptom looks like undertraining rather than a units bug."
    )

    mean, log_std = gaussian_head(layers, ref["states"], log_std_min, log_std_max)
    action = squashed_action(
        layers,
        ref["states"],
        deterministic=True,
        action_scale=scale,
        action_bias=bias,
        log_std_min=log_std_min,
        log_std_max=log_std_max,
    )

    d_mean = float(np.max(np.abs(mean - ref["mean"])))
    d_log_std = float(np.max(np.abs(log_std - ref["log_std"])))
    d_action = float(np.max(np.abs(action - ref["action"])))

    assert d_mean < GAUSSIAN_TOLERANCE, (
        f"[Pendulum-v1] max |Δ pre-squash mean| = {d_mean:.3e}, tolerance "
        f"{GAUSSIAN_TOLERANCE:.0e}. A mean mismatch is a WEIGHT problem — "
        "transposed matrix or missing bias. See api/forward.py."
    )
    assert d_log_std < GAUSSIAN_TOLERANCE, (
        f"[Pendulum-v1] max |Δ log σ| = {d_log_std:.3e} with a correct mean. That "
        "combination is the CLAMP: train/nets.py clamps to "
        f"[{log_std_min}, {log_std_max}] and api/forward.gaussian_head must apply "
        "the identical clamp. A clamp applied on one side only is a different "
        "function, not a safety net."
    )
    assert d_action < ACTION_TOLERANCE, (
        f"[Pendulum-v1] max |Δ action| = {d_action:.3e} with a correct mean and "
        "log σ. That combination is the ACTION SCALE — check that action_scale=2 "
        "made it into the archive."
    )
    print(
        f"[Pendulum-v1] max |Δ mean| = {d_mean:.3e}, max |Δ log σ| = {d_log_std:.3e}, "
        f"max |Δ action| = {d_action:.3e}"
    )

    # And the log σ clamp must actually have engaged somewhere in this batch, or
    # the assertion above agreed about a clamp that was never applied. The
    # reference observations are N(0, 3) precisely so that it does.
    assert np.any(np.isclose(log_std, log_std_max)) or float(log_std.max()) > 0.0, (
        "no observation in the reference batch drove log σ anywhere near its "
        "ceiling, so this test did not exercise the clamp. Widen the reference "
        "observations in train/nets.dump_reference."
    )


def test_a_transposed_export_is_actually_caught(reference):
    """Deliberately break it the usual way, and confirm the test would notice.

    A test that has never failed is a test you cannot trust. This one corrupts
    the weights exactly as a wrong export would and asserts that the comparison
    above would have caught it — which is what licences the claim that a green
    suite means the artifacts are right.

    The middle layer is square (64×64), so transposing it survives every shape
    check: the arithmetic still runs and the answers are silently wrong. That is
    precisely why the real bug is so hard to find by inspection.
    """
    for stem in ("cartpole", "acrobot"):
        _z, layers = _layers(reference, stem)
        ref = np.load(reference / f"reference_{stem}.npz")
        W1, b1 = layers[1]
        layers[1] = (W1.T.copy(), b1)
        broken = action_probabilities(layers, ref["states"])
        moved = float(np.max(np.abs(broken - ref["probs"])))
        assert moved > PROB_TOLERANCE, (
            f"[{stem}] transposing a hidden layer did not move the output "
            "distribution by more than the tolerance, which means the tolerance "
            "is too loose to catch the bug it exists for"
        )


def test_dropping_the_action_scale_is_actually_caught(reference):
    """The SAC-specific version of the test above.

    Serving `tanh(mean)` instead of `2·tanh(mean)` is the mistake this product's
    continuous head invites, and it is invisible in every shape check. Confirm
    the tolerance is tight enough to see it.
    """
    z, layers = _layers(reference, "pendulum")
    ref = np.load(reference / "reference_pendulum.npz")
    unscaled = squashed_action(layers, ref["states"], deterministic=True, action_scale=1.0)
    moved = float(np.max(np.abs(unscaled - ref["action"])))
    assert moved > ACTION_TOLERANCE, (
        "dropping action_scale did not change the action by more than the "
        "tolerance — which would mean either the tolerance is useless or the "
        "reference policy outputs zero torque everywhere"
    )
    assert float(np.asarray(z["action_scale"]).reshape(-1)[0]) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# The artifacts actually sitting in policies/
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "env_id", "obs_dim", "n_actions", "space"),
    [
        ("a2c_cartpole", "CartPole-v1", 4, 2, "discrete"),
        ("ppo_acrobot", "Acrobot-v1", 6, 3, "discrete"),
        ("sac_pendulum", "Pendulum-v1", 3, 1, "continuous"),
    ],
)
def test_the_deployed_artifacts_load_with_the_shapes_their_envs_have(
    name, env_id, obs_dim, n_actions, space
):
    """The committed artifacts must satisfy the layout the forward pass assumes.

    Distinct from the tests above: those check freshly generated policies, this
    one checks the files actually sitting in policies/ that the service will
    load in production. A mismatch here means every well-formed request to that
    policy gets a 422 — the endpoint works perfectly and rejects everything.
    """
    store = PolicyArtifactStore("policies", default_name="a2c_cartpole")
    policy, meta = store.get(name)
    assert meta["kind"] == "mlp"
    assert meta["env_id"] == env_id, (
        f"{name} does not record env_id={env_id}; /rollout cannot build its environment"
    )
    assert meta["obs_dim"] == obs_dim and meta["n_actions"] == n_actions, (
        f"the deployed {name} artifact does not match {env_id}'s shapes "
        f"({meta['obs_dim']}, {meta['n_actions']}); /act will reject every "
        "well-formed request with a 422"
    )
    assert meta["action_space"] == space

    action, value = policy.act(np.zeros(obs_dim))
    if space == "discrete":
        assert isinstance(action, int) and 0 <= action < n_actions
        p = policy.probabilities(np.zeros(obs_dim))
        assert p.shape == (n_actions,) and np.isclose(p.sum(), 1.0)
    else:
        assert isinstance(action, list) and len(action) == n_actions
        assert abs(action[0]) <= 2.0 + 1e-9, "a tanh-squashed action cannot leave [-2, 2]"
        assert np.isfinite(value)
