"""
THE REQUIRED TEST OF THIS TOPIC: the NumPy forward pass must reproduce the
PyTorch action distribution.

Everything else in this repository can be right while this is wrong, and the
symptom is an agent that scored 480 in training and behaves randomly in the
deployed app — with no error anywhere, because a transposed matrix is still a
matrix and NumPy will happily multiply it. That failure has cost more student
weekends than any other in this course, so it is a test rather than a step in a
checklist.

TOLERANCE: 1e-5 on the maximum absolute difference between the two probability
vectors, over 256 observations drawn wide enough to saturate the softmax.

That number is chosen, not copied. `train/export.py` writes float32 weights and
`api/forward.py` evaluates them in float64, so the two implementations differ by
accumulated rounding of order 1e-7 on a 4-64-64-2 network. 1e-5 is two orders of
magnitude above the noise and several orders below any real bug: a transposed
weight matrix, a missing bias or a doubled softmax all move probabilities by
1e-2 or more. A tolerance of 1e-2 would pass while broken; 1e-9 would fail on a
different CPU.

WHEN IT FAILS, in order of how often each is the cause — the diagnostic recipe
is spelled out at the bottom of api/forward.py:

    1. A transposed weight matrix. PyTorch stores nn.Linear.weight as
       (out_features, in_features) and export.py writes it out UNCHANGED.
       Print W0.shape and policy.net[0].weight.shape; they must be identical,
       not transposes of each other.
    2. A missing bias — a layer built with bias=False, so `b{i}` is absent, so
       `layers_from_npz` stops early and deploys a shorter network in silence.
    3. A softmax or ReLU on the output layer of the PyTorch module, applied
       twice on one side and once on the other.
    4. Observation normalisation applied on one side only.
    5. Only then, precision.

Note that this test compares DISTRIBUTIONS, not actions. Comparing argmaxes
would pass for a badly wrong network whenever the error did not happen to flip
the winner, and with two actions that is most of the time.
"""

from __future__ import annotations

import numpy as np
import pytest

from api.forward import action_probabilities, layers_from_npz
from api.policy import PolicyArtifactStore

TOLERANCE = 1e-5


@pytest.fixture(scope="module")
def reference(tmp_path_factory, request):
    """Export a seeded policy and its PyTorch probabilities from a subprocess.

    The subprocess is the point — see `run_torch_script` in conftest.py. This
    test process must remain free of torch or tests/test_no_torch.py becomes a
    lie.
    """
    request.getfixturevalue("requires_torch")
    from tests.conftest import run_torch_script

    out = tmp_path_factory.mktemp("equivalence")
    run_torch_script(
        f"""
        from train.policy import dump_reference
        print(json.dumps(dump_reference({str(out)!r}, seed=0, n=256)))
        """
    )
    ref = np.load(out / "reference.npz")
    z = np.load(out / "equivalence_policy.npz", allow_pickle=False)
    return {"states": ref["states"], "probs": ref["probs"], "layers": layers_from_npz(z)}


def test_numpy_reproduces_the_torch_action_distribution(reference):
    ours = action_probabilities(reference["layers"], reference["states"])
    theirs = reference["probs"]

    assert ours.shape == theirs.shape, (
        f"shape mismatch {ours.shape} vs {theirs.shape} — the NumPy forward pass "
        "is not even producing one distribution per observation"
    )

    max_abs_diff = float(np.max(np.abs(ours - theirs)))
    assert max_abs_diff < TOLERANCE, (
        f"max |NumPy - PyTorch| = {max_abs_diff:.3e}, tolerance {TOLERANCE:.0e}.\n"
        "This is almost always a TRANSPOSED WEIGHT MATRIX or a MISSING BIAS. Check:\n"
        "    python -c \"import numpy as np; z=np.load('policies/vpg_cartpole.npz'); \"\n"
        "             \"print({k: z[k].shape for k in z.files})\"\n"
        "and compare each W{i}.shape against the corresponding\n"
        "    policy.net[2*i].weight.shape\n"
        "in PyTorch. They must MATCH, not transpose. See api/forward.py."
    )
    # Printed so the measured number lands in the report rather than only the
    # verdict. "It passed" is weaker evidence than "it agreed to 3e-8".
    print(f"max |NumPy - PyTorch| = {max_abs_diff:.3e} (tolerance {TOLERANCE:.0e})")


def test_the_probabilities_are_a_distribution(reference):
    """Rows sum to 1 and nothing is NaN.

    A separate assertion because the difference test can pass while both
    implementations are broken in the same direction — if the exported weights
    were all zeros, both would return [0.5, 0.5] everywhere and agree perfectly.
    """
    ours = action_probabilities(reference["layers"], reference["states"])
    assert np.all(np.isfinite(ours))
    assert np.allclose(ours.sum(axis=-1), 1.0, atol=1e-9)
    assert ours.min() >= 0.0
    # And it must not be the degenerate all-0.5 policy that would make the
    # agreement above meaningless.
    assert float(np.abs(ours - 0.5).max()) > 1e-3, (
        "every probability is 0.5, so this test agreed about nothing. Did the "
        "export write zeros?"
    )


def test_a_transposed_export_is_actually_caught(reference):
    """Deliberately break it the usual way, and confirm the test would notice.

    A test that has never failed is a test you cannot trust. This one corrupts
    the weights exactly as a wrong export would and asserts that the comparison
    above would have caught it — which is what licences the claim that a green
    suite means the artifact is right.
    """
    layers = list(reference["layers"])
    W0, b0 = layers[0]
    # Square-ish middle layers make a transpose survive the shape check, which
    # is precisely why the real bug is silent. Here we transpose the second
    # layer's 64x64 matrix: shapes still line up, arithmetic still runs,
    # answers are wrong.
    W1, b1 = layers[1]
    layers[1] = (W1.T.copy(), b1)
    broken = action_probabilities(layers, reference["states"])
    max_abs_diff = float(np.max(np.abs(broken - reference["probs"])))
    assert max_abs_diff > TOLERANCE, (
        "transposing a hidden layer did not move the output distribution by more "
        "than the tolerance, which means the tolerance is too loose to catch the "
        "bug it exists for"
    )
    assert W0.shape[1] == 4 and b0.shape == (W0.shape[0],)


def test_the_deployed_artifact_loads_and_agrees_with_its_own_shapes():
    """The committed artifact must satisfy the layout the forward pass assumes.

    Distinct from the test above: that one checks a freshly generated policy,
    this one checks the file actually sitting in policies/ that the service will
    load in production.
    """
    store = PolicyArtifactStore("policies", default_name="vpg_cartpole")
    policy, meta = store.get("vpg_cartpole")
    assert meta["kind"] == "mlp"
    assert meta["obs_dim"] == 4 and meta["n_actions"] == 2, (
        "the deployed artifact does not match CartPole-v1's shapes; /act will "
        "reject every well-formed request with a 422"
    )
    p = policy.probabilities(np.array([0.0, 0.0, 0.05, 0.0]))
    assert p.shape == (2,)
    assert np.isclose(p.sum(), 1.0)
