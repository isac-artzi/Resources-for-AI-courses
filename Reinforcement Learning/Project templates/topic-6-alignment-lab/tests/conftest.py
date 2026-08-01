"""
Shared fixtures. Note that every test drives the app through an HTTP test
client rather than by calling handler functions — the course quality bar
requires it, and the reason is that half the contract (status codes,
validation, serialisation) only exists at the HTTP boundary.

Topic 6 carries forward the subprocess machinery Topic 3 introduced, and needs
it more than any previous topic: the equivalence test has to run PyTorch AND
scikit-learn to produce a reference score, while this process must remain free
of torch because `tests/test_no_torch.py` asserts exactly that. See
`run_torch_script`.
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

# The stub vocabulary, named at module scope so tests can DETECT the stub and
# skip results that would be meaningless against it. A test that silently
# passes against a random-weight artifact is worse than a skipped one: it
# reports green for a model that does not exist.
STUB_VOCAB = ["specific", "evidence", "tested", "vague", "hype", "whatever"]


@pytest.fixture(scope="session", autouse=True)
def _ensure_artifact():
    """Guarantee loadable artifacts so the suite runs on a fresh clone.

    Two artifacts, because this topic serves two kinds:

      * `smoke_test_policy.npz` — the standing 4-state / 2-action Q-table the
        base suite's /act tests drive.
      * `reward_tfidf.npz` — a TINY reward head with a six-word vocabulary,
        written only if you have not yet run `python -m train.train`. It is
        enough for the schema, routing and audit tests to exercise the real code
        path; it has learned nothing and proves nothing about the model.

    The real head, once exported, sits at the same path and is picked up
    automatically. The tests that are about the MODEL rather than about the
    plumbing take the `trained_head_path` fixture, which skips on the stub.
    """
    policies = ROOT / "policies"
    policies.mkdir(exist_ok=True)

    smoke = policies / "smoke_test_policy.npz"
    if not smoke.exists():
        np.savez_compressed(smoke, Q=np.eye(4, 2, dtype=np.float32))

    head = policies / "reward_tfidf.npz"
    if not head.exists():
        vocab = np.asarray(STUB_VOCAB, dtype=np.str_)
        rng = np.random.default_rng(0)
        np.savez_compressed(
            head,
            vocab=vocab,
            idf=np.ones(len(vocab), dtype=np.float32),
            W0=rng.normal(scale=0.5, size=(8, len(vocab))).astype(np.float32),
            b0=np.zeros(8, dtype=np.float32),
            W1=rng.normal(scale=0.5, size=(1, 8)).astype(np.float32),
            b1=np.zeros(1, dtype=np.float32),
        )
    yield


def artifact_is_stub(path: pathlib.Path) -> bool:
    z = np.load(path, allow_pickle=False)
    return "vocab" in z.files and list(np.asarray(z["vocab"]).astype(str)) == STUB_VOCAB


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="session")
def trained_head_path():
    """Path to the deployed reward head, skipping if it is the conftest stub."""
    path = ROOT / "policies" / "reward_tfidf.npz"
    if not path.exists() or artifact_is_stub(path):
        pytest.skip(
            "policies/reward_tfidf.npz is the conftest stub. Run "
            "`python -m train.train --offline --quick` first — this test is "
            "about the model, and the stub has not learned anything."
        )
    return path


@pytest.fixture(scope="session")
def reports_dir():
    """`reports/`, skipping if the pipeline has not been run.

    Tests that assert on measured RESULTS read them from here rather than
    recomputing them. Recomputing would double the pipeline's runtime inside
    the test suite and — worse — would let the test pass against numbers the
    README never saw.
    """
    path = ROOT / "reports"
    if not (path / "reward_heads.json").exists():
        pytest.skip("no reports/reward_heads.json — run `python -m train.train --offline --quick`")
    return path


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

    scikit-learn is probed alongside torch because this topic's equivalence
    test needs BOTH: the reference score is a scikit-learn TF-IDF transform fed
    into a PyTorch module, and it is the whole chain that must agree.
    """
    probe = subprocess.run(
        [sys.executable, "-c", "import torch, sklearn"], capture_output=True, text=True
    )
    return probe.returncode == 0


@pytest.fixture(scope="session")
def requires_torch(torch_available):
    if not torch_available:
        pytest.skip(
            "torch and/or scikit-learn are not installed (expected in CI, which "
            "installs serving deps only)"
        )
