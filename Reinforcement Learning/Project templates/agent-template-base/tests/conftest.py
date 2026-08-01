"""
Shared fixtures. Note that every test drives the app through an HTTP test
client rather than by calling handler functions — the course quality bar
requires it, and the reason is that half the contract (status codes,
validation, serialisation) only exists at the HTTP boundary.
"""

from __future__ import annotations

import pathlib
import sys

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


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    return TestClient(app, raise_server_exceptions=False)
