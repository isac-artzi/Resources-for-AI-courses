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

    # TOPIC 1 ADDS A SECOND ONE. The table above has four states and this
    # topic's environment has sixty-four, so /rollout against the real lake
    # would ask it for a state index it does not have and fail for a reason
    # that says nothing about the endpoint under test. This artifact is the
    # right shape and untrained: every entry is the SAME value, so the greedy
    # action ties to 0 everywhere and the softmax over any row is uniform —
    # which is what the rollout tests assert on. The rollout test is about the
    # wiring between the service tier and the environment, not about a score.
    #
    # The constant is 0.25 rather than 0.0 for one non-obvious reason: an
    # all-zero 64x4 float32 archive is byte-for-byte identical to the
    # `untrained_policy.npz` that train/train.py exports, identical bytes mean
    # an identical SHA-256, and /policies de-duplicates by checksum. This
    # fixture would then hide the artifact the "Watch" tab goes looking for.
    lake = policies / "lake_smoke_policy.npz"
    if not lake.exists():
        np.savez_compressed(lake, Q=np.full((64, 4), 0.25, dtype=np.float32))
    yield


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    return TestClient(app, raise_server_exceptions=False)
