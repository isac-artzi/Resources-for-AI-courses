"""
Shared fixtures. Note that every test drives the app through an HTTP test client
rather than by calling handler functions — the course quality bar requires it,
and the reason is that half the contract (status codes, validation,
serialisation) only exists at the HTTP boundary.

This topic carries one piece of machinery forward from Topic 3 that is worth
understanding before you copy it: `run_torch_script`. The three equivalence
tests need PyTorch, and this test process is required to stay free of it —
`tests/test_no_torch.py` asserts exactly that. See the docstring below.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The three deployed agents, with the shapes their environments actually have.
# Written down here as well as in envs/__init__.py on purpose: a test that
# imports the same constant the code under test uses cannot catch a wrong
# constant. These are read off the Gymnasium documentation, independently.
AGENTS = {
    "a2c_cartpole": {"env_id": "CartPole-v1", "obs_dim": 4, "n_actions": 2, "continuous": False},
    "ppo_acrobot": {"env_id": "Acrobot-v1", "obs_dim": 6, "n_actions": 3, "continuous": False},
    "sac_pendulum": {"env_id": "Pendulum-v1", "obs_dim": 3, "n_actions": 1, "continuous": True},
}


@pytest.fixture(scope="session", autouse=True)
def _ensure_artifacts():
    """Guarantee loadable artifacts so the suite runs on a fresh clone.

    The REAL artifacts — the ones `python -m train.train` exported and you
    committed — are what these tests should run against, and they are picked up
    automatically because they sit in the same directory. This fallback exists
    only so that a fresh fork's suite is green before the first training run has
    finished. It writes random weights, so it proves the plumbing and nothing
    whatever about the agents.
    """
    policies = ROOT / "policies"
    policies.mkdir(exist_ok=True)

    smoke = policies / "smoke_test_policy.npz"
    if not smoke.exists():
        np.savez_compressed(smoke, Q=np.eye(4, 2, dtype=np.float32))

    rng = np.random.default_rng(0)
    for name, spec in AGENTS.items():
        path = policies / f"{name}.npz"
        if path.exists():
            continue
        # A squashed-Gaussian actor emits 2 * action_dim outputs: mean and
        # log σ. Getting this wrong in the fallback would make the fallback
        # itself a different shape from the real artifact, and the shape tests
        # would pass on a fresh clone and fail after training — which is the
        # worst possible ordering.
        out = spec["n_actions"] * (2 if spec["continuous"] else 1)
        arrays = {
            "W0": rng.normal(scale=0.3, size=(64, spec["obs_dim"])).astype(np.float32),
            "b0": np.zeros(64, dtype=np.float32),
            "W1": rng.normal(scale=0.3, size=(64, 64)).astype(np.float32),
            "b1": np.zeros(64, dtype=np.float32),
            "W2": rng.normal(scale=0.3, size=(out, 64)).astype(np.float32),
            "b2": np.zeros(out, dtype=np.float32),
            "head": np.asarray(
                "squashed_gaussian" if spec["continuous"] else "categorical"
            ),
            "env_id": np.asarray(spec["env_id"]),
        }
        if spec["continuous"]:
            arrays |= {
                "action_scale": np.asarray([2.0], dtype=np.float32),
                "action_bias": np.asarray([0.0], dtype=np.float32),
                "log_std_min": np.asarray([-20.0], dtype=np.float32),
                "log_std_max": np.asarray([2.0], dtype=np.float32),
            }
        np.savez_compressed(path, **arrays)
    yield


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Running the training tier from the test suite, without importing it
# ---------------------------------------------------------------------------


def run_torch_script(body: str, timeout: int = 900) -> dict:
    """Execute `body` in a FRESH interpreter and return the JSON it printed last.

    Why a subprocess instead of `import torch` at the top of the test module:

    `sys.modules` is per-process. One `import torch` anywhere in this suite
    leaves "torch" in `sys.modules` for every test that runs afterwards,
    including `tests/test_no_torch.py`, whose whole job is to assert that it is
    absent. The guard protecting the 690 MB deployment budget would start
    failing for a reason that has nothing to do with the deployment, pytest
    collection order would decide whether the build was green, and the obvious
    "fix" would be to weaken the guard. Paying one interpreter spawn per test
    module is much cheaper than any of that.

    It also mirrors reality: the training tier IS a separate process from the
    serving tier. A test that reaches into it by import is testing an
    arrangement you never deploy.

    The script must print exactly one JSON object as its LAST line of stdout;
    anything before it (progress logging, warnings) is ignored, which is what
    makes it safe to leave a `print` in while debugging.
    """
    script = textwrap.dedent(
        f"""
        import json, sys
        sys.path.insert(0, {str(ROOT)!r})
        {textwrap.indent(textwrap.dedent(body), "        ").lstrip()}
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(ROOT),
    )
    if proc.returncode != 0:
        raise AssertionError(
            "the training-tier subprocess failed:\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    if not lines:
        raise AssertionError(f"the subprocess printed nothing.\nstderr:\n{proc.stderr}")
    return json.loads(lines[-1])


@pytest.fixture(scope="session")
def torch_available() -> bool:
    """Is the TRAINING environment installed here?

    CI installs `requirements-serve.txt` and nothing else, on purpose — see
    .github/workflows/ci.yml — so torch is genuinely absent there and the tests
    that need it skip. That is not a hole in the coverage: those tests run on
    your machine and in the training job, and the tests CI does run are the ones
    that protect the deployment. Never make an equivalence test pass by stubbing
    torch out; a stub cannot disagree with NumPy, which is the entire thing the
    test is looking for.
    """
    probe = subprocess.run(
        [sys.executable, "-c", "import torch"], capture_output=True, text=True
    )
    return probe.returncode == 0


@pytest.fixture(scope="session")
def requires_torch(torch_available):
    if not torch_available:
        pytest.skip("torch is not installed (expected in CI, which installs serving deps only)")
