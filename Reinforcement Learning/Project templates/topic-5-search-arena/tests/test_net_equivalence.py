"""
The exported NumPy network must reproduce the PyTorch one it came from.

Everything else in this repository can be right while this is wrong, and the
symptom is an agent that improved steadily through self-play and then plays
badly in the deployed app — with no error anywhere, because a transposed matrix
is still a matrix and NumPy will happily multiply it.

TOLERANCE: 1e-5 on the maximum absolute difference in the logits and in the
value. That number is chosen, not copied. `train/export.py` writes float32 and
`search/net.py` evaluates in float64, so the two differ by accumulated rounding
of order 1e-7 on an 84-128-128 network. 1e-5 is two orders of magnitude above
the noise and several below any real bug: a transposed weight, a missing bias or
a doubled activation all move logits by 1e-2 or more.

WHEN IT FAILS, in order of how often each is the cause:

    1. A transposed weight matrix. `nn.Linear.weight` is (out, in) and
       `_extract_arrays` writes it out UNCHANGED, because `search/net.py`
       computes `W @ x + b`. Print the two shapes; they must be identical, not
       transposes of each other.
    2. A missing head. An archive with `Wp` but no `Wv` loads as a plain MLP in
       `api/policy.py` and serves a 128-dimensional "action" in silence.
    3. An activation applied on one side only — the value head's `tanh` is the
       usual one, because it lives in `forward()` on the torch side and in
       `PolicyValueNet.forward` on the NumPy side.
    4. Only then, precision.

The test runs the torch half in a SUBPROCESS. See `run_torch_script` in
conftest.py: one `import torch` in this process would make
`tests/test_no_torch.py` fail for a reason that has nothing to do with the
deployment.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from search.net import PolicyValueNet
from tests.conftest import run_torch_script

TOLERANCE = 1e-5


@pytest.fixture(scope="module")
def exported(tmp_path_factory, torch_available):
    """Build, export and evaluate a seeded network in the training tier."""
    if not torch_available:
        pytest.skip("torch is not installed here — this is expected in CI")
    out = tmp_path_factory.mktemp("net") / "equivalence.npz"
    payload = run_torch_script(
        f"""
        import numpy as np, torch
        from train.selfplay import OBS_DIM, build_network, _extract_arrays
        from train.export import export_policy_value_net

        module = build_network(seed=1234)
        module.eval()
        arrays = _extract_arrays(module)
        export_policy_value_net(arrays, {str(out)!r})

        rng = np.random.default_rng(0)
        # Dense random inputs, not real boards. A real Connect Four position is
        # sparse and mostly zeros, which will happily hide a transposed weight
        # in a layer whose input happens to be symmetric.
        xs = rng.normal(size=(64, OBS_DIM)).astype(np.float32)
        with torch.no_grad():
            logits, value = module(torch.from_numpy(xs))
        print(json.dumps({{
            "xs": xs.tolist(),
            "logits": logits.numpy().tolist(),
            "value": value.numpy().ravel().tolist(),
        }}))
        """
    )
    return pathlib.Path(out), payload


def test_numpy_forward_matches_torch(exported):
    path, payload = exported
    net = PolicyValueNet.from_npz(path)
    xs = np.asarray(payload["xs"], dtype=np.float64)

    worst_logit = worst_value = 0.0
    for i, x in enumerate(xs):
        logits, value = net.forward(x)
        worst_logit = max(worst_logit, float(np.max(np.abs(logits - payload["logits"][i]))))
        worst_value = max(worst_value, abs(value - payload["value"][i]))

    assert worst_logit < TOLERANCE, (
        f"logits differ by {worst_logit:.2e}. See this module's docstring for the "
        "ordered list of causes; a transposed weight is the first one to check."
    )
    assert worst_value < TOLERANCE, f"value differs by {worst_value:.2e}"


def test_the_archive_declares_its_shape(exported):
    path, _ = exported
    net = PolicyValueNet.from_npz(path)
    assert net.obs_dim == 84, "two 42-cell planes — see shared/preprocess.py"
    assert net.n_actions == 7


def test_an_archive_missing_a_head_is_refused(tmp_path):
    """A loader that silently accepts an archive it does not understand is worse
    than one that refuses it. `api/policy.py` would otherwise load W0/b0/W1/b1 as
    a plain MLP and serve a 128-dimensional 'action'."""
    path = tmp_path / "half.npz"
    np.savez_compressed(
        path,
        W0=np.zeros((8, 84), dtype=np.float32), b0=np.zeros(8, dtype=np.float32),
        W1=np.zeros((8, 8), dtype=np.float32), b1=np.zeros(8, dtype=np.float32),
        Wp=np.zeros((7, 8), dtype=np.float32), bp=np.zeros(7, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="Wv"):
        PolicyValueNet.from_npz(path)


def test_priors_are_masked_to_legal_moves():
    """PUCT must never spend exploration on a column that cannot be played.

    Checked against the COMMITTED artifact (or the conftest fallback), because
    this is a property of the serving path rather than of any particular weights.
    """
    from envs.connect_four import COLS, Position

    net = PolicyValueNet.from_npz(pathlib.Path("policies/alphazero_c4.npz"))
    position = Position()
    # Both sides play column 0 six times. Because the players alternate it fills
    # as Y R Y R Y R — full, with nobody having four, so the game is still live.
    for _ in range(6):
        position.push(0)
    assert not position.is_terminal() and 0 not in position.legal_moves()
    prior, value = net.evaluate(position)
    assert prior.shape == (COLS,)
    assert prior[0] == 0.0, "a full column must receive zero prior mass"
    assert prior.sum() == pytest.approx(1.0)
    assert -1.0 <= value <= 1.0, "the value head is tanh-squashed"
