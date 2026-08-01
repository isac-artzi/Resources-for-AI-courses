"""
The data-tier round trip: write an episode row, read it back, and check the
aggregate the run-history view depends on.

Against a configured project this exercises Supabase. In CI, with no
credentials, it exercises the in-process fallback — the SAME interface and the
same row shapes, so a schema mistake still surfaces. What it cannot catch is a
Postgres-side error (a bad constraint, a missing index); apply the migration
and run this against your real project at least once before submitting.
"""

from __future__ import annotations

import pytest

from shared.config import get_settings
from shared.store import get_store


def test_episode_write_read_roundtrip():
    store = get_store()
    experiment_id = store.insert_experiment(
        {"algorithm": "test", "env_id": "test-env", "seed": 0, "hyperparameters": {"alpha": 0.1}}
    )
    rows = [
        {"experiment_id": experiment_id, "episode_index": i, "return": float(i % 2), "length": 10,
         "epsilon": 0.1}
        for i in range(25)
    ]
    assert store.insert_episodes(rows) == 25

    read_back = store.episodes_for(experiment_id)
    assert len(read_back) == 25
    assert read_back[0]["episode_index"] == 0
    assert "epsilon" in read_back[0], "epsilon must be logged per episode, not per run"


def test_run_summary_aggregates():
    store = get_store()
    experiment_id = store.insert_experiment(
        {"algorithm": "test-agg", "env_id": "test-env", "seed": 7, "hyperparameters": {}}
    )
    store.insert_episodes(
        [{"experiment_id": experiment_id, "episode_index": i, "return": 1.0, "length": 5}
         for i in range(10)]
    )
    store.insert_evaluation(
        {"experiment_id": experiment_id, "at_training_episode": 10, "episodes": 20,
         "mean_return": 0.8, "std_return": 0.1, "stderr_return": 0.0224}
    )
    runs = store.recent_runs(50)
    mine = [r for r in runs if r["experiment_id"] == experiment_id]
    assert mine, "the run we just wrote should appear in the run history"
    assert mine[0]["episodes_logged"] == 10
    assert mine[0]["eval_mean_return"] == pytest.approx(0.8)


def test_runs_endpoint_flags_degradation_rather_than_returning_an_empty_table(client):
    body = client.get("/runs").json()
    assert "degraded" in body, "the UI must be able to distinguish 'no runs' from 'no database'"
    if not get_settings().data_tier_configured:
        assert body["degraded"] is False  # the fallback IS reachable; it is just local
