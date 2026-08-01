"""
Shared fixtures. Note that every test drives the app through an HTTP test
client rather than by calling handler functions — the course quality bar
requires it, and the reason is that half the contract (status codes,
validation, serialisation) only exists at the HTTP boundary.

Topic 5 adds two pieces of machinery worth understanding before you copy them:

  * `_ensure_value_net` writes a randomly initialised policy-value archive if
    none exists, so a fresh fork's suite is green before the first training run.
    It is random weights: it proves the plumbing and nothing about the agent.
  * `run_torch_script` runs training-tier code in a SUBPROCESS. One `import
    torch` anywhere in this process would leave "torch" in `sys.modules` for
    every test that runs afterwards, including `tests/test_no_torch.py`, whose
    whole job is to assert that it is absent.
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


@pytest.fixture(scope="session", autouse=True)
def _ensure_artifact():
    """Guarantee at least one loadable artifact so the suite runs on a fresh clone.

    A fixture that creates a trivial 4-state / 2-action table. Your own trained
    artifact, once exported, sits alongside it and is picked up automatically.
    """
    policies = ROOT / "policies"
    policies.mkdir(exist_ok=True)
    smoke = policies / "smoke_test_policy.npz"
    if not smoke.exists():
        np.savez_compressed(smoke, Q=np.eye(4, 2, dtype=np.float32))
    yield


@pytest.fixture(scope="session", autouse=True)
def _ensure_value_net(_ensure_artifact):
    """A policy-value archive so the learned agent exists in the registry.

    The REAL artifact — the one `python -m train.train` exported and you
    committed — is what these tests should run against, and it is picked up
    automatically if it is there. This fallback is random weights and exists
    only so that a fresh fork's suite is green before the first training run
    finishes. It proves the archive loads and the PUCT path runs; it proves
    nothing about play strength, and no test in this suite asserts that the
    learned agent is strong on the basis of it.
    """
    path = ROOT / "policies" / "alphazero_c4.npz"
    if not path.exists():
        rng = np.random.default_rng(0)
        np.savez_compressed(
            path,
            **{
                "W0": rng.normal(scale=0.1, size=(128, 84)).astype(np.float32),
                "b0": np.zeros(128, dtype=np.float32),
                "W1": rng.normal(scale=0.1, size=(128, 128)).astype(np.float32),
                "b1": np.zeros(128, dtype=np.float32),
                "Wp": rng.normal(scale=0.1, size=(7, 128)).astype(np.float32),
                "bp": np.zeros(7, dtype=np.float32),
                "Wv": rng.normal(scale=0.1, size=(1, 128)).astype(np.float32),
                "bv": np.zeros(1, dtype=np.float32),
            },
        )
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
    "fix" would be to weaken the guard. Paying one interpreter spawn per test is
    much cheaper than any of that.

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
    that protect the deployment. Never make the equivalence test pass by
    stubbing torch out; a stub cannot disagree with NumPy, which is the entire
    thing the test is looking for.
    """
    probe = subprocess.run(
        [sys.executable, "-c", "import torch"], capture_output=True, text=True
    )
    return probe.returncode == 0
