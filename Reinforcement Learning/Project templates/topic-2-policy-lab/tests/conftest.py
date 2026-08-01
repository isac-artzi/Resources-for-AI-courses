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


@pytest.fixture(scope="session", autouse=True)
def _ensure_topic2_artifacts(_ensure_artifact):
    """Guarantee both policy sources exist before any test touches the service.

    `policies/value_iteration.npz` and `policies/monte_carlo.npz` are COMMITTED
    — they are what the deployed app serves, and a repository whose tests only
    pass after a training run is a repository whose CI is really a training
    job. This fixture is the safety net for the case where a student has
    deleted them, and it deliberately uses a tiny Monte Carlo budget: its job
    is to make the routing tests runnable, not to reproduce a result.
    """
    from envs import make_env
    from train.export import export_tabular_policy
    from train.monte_carlo import first_visit_mc_evaluation, mc_control_exploring_starts
    from train.value_iteration import value_iteration

    policies = ROOT / "policies"
    env = make_env()
    core = env.unwrapped

    if not (policies / "value_iteration.npz").exists():
        plan = value_iteration(core.P, core.n_states, core.n_actions)
        export_tabular_policy(plan.Q, plan.V, plan.policy,
                              policies / "value_iteration.npz")
    if not (policies / "monte_carlo.npz").exists():
        control = mc_control_exploring_starts(env, episodes=5_000, collect_rows=False)
        ev = first_visit_mc_evaluation(env, control.policy, episodes=2_000,
                                       seed=1, collect_rows=False)
        export_tabular_policy(control.Q, ev.V, control.policy,
                              policies / "monte_carlo.npz")
    yield


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    return TestClient(app, raise_server_exceptions=False)
