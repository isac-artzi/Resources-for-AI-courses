"""
The `gradient_stats` round trip: write update rows, read them back, and check
that the shape the Gradient Variance tab depends on survives the journey.

Against a configured project this exercises Supabase. In CI, with no
credentials, it exercises the in-process fallback — the SAME interface and the
same row shapes, so a column you forgot to add to 002_gradient_stats.sql still
shows up here as a Pydantic validation error. What it cannot catch is a
Postgres-side error (the unique constraint, the foreign key, the jsonb cast in
`gradient_variance_by_arm`); apply the migration and run this against your real
project at least once before submitting.
"""

from __future__ import annotations

import numpy as np
import pytest

from shared.schemas import GradientStatRow
from shared.store import get_store


def _row(experiment_id: str, i: int, **over) -> dict:
    row = {
        "experiment_id": experiment_id,
        "update_index": i,
        "episode_index": (i + 1) * 10,
        "gradient_norm": 1.0 + i,
        "gradient_variance": 100.0 / (i + 1),
        "policy_entropy": 0.69 - 0.01 * i,
        "off_policy": False,
    }
    return row | over


def test_gradient_stats_write_read_roundtrip():
    store = get_store()
    experiment_id = store.insert_experiment(
        {"algorithm": "vpg+baseline", "env_id": "CartPole-v1", "seed": 0,
         "hyperparameters": {"use_baseline": True, "use_importance_sampling": False}}
    )
    rows = [_row(experiment_id, i) for i in range(8)]
    assert store.insert_gradient_stats(rows) == 8

    read_back = store.gradient_stats_for(experiment_id)
    assert len(read_back) == 8
    assert [r["update_index"] for r in read_back] == list(range(8)), (
        "rows must come back in update order — the chart plots them as a line and "
        "an unordered read draws a scribble"
    )
    assert read_back[0]["episode_index"] == 10, (
        "episode_index is what lets the variance chart share an x-axis with the "
        "learning curve; without it the headline chart cannot be drawn"
    )


def test_every_row_validates_against_the_schema():
    """The row dict and the Pydantic model must agree, or the endpoint 500s.

    This is the cheap version of the migration check: shared/schemas.py and
    db/migrations/002_gradient_stats.sql are two descriptions of one table, and
    they drift the moment you add a column to one of them at 2 a.m.
    """
    store = get_store()
    experiment_id = store.insert_experiment(
        {"algorithm": "vpg+is", "env_id": "CartPole-v1", "seed": 1, "hyperparameters": {}}
    )
    store.insert_gradient_stats([_row(experiment_id, 0)])
    for r in store.gradient_stats_for(experiment_id):
        GradientStatRow(**r)  # raises on a missing or mistyped field


def test_off_policy_rows_carry_the_importance_weight_distribution():
    """An off-policy row without weight statistics is a row that cannot be plotted.

    The assignment asks for the DISTRIBUTION of importance weights, so the
    histogram is part of the contract and not an extra. Storing a histogram
    rather than the raw weights is a size decision — one run at the real budget
    produces a few hundred thousand of them — and this test pins the shape the
    UI reads.
    """
    store = get_store()
    experiment_id = store.insert_experiment(
        {"algorithm": "vpg+baseline+is", "env_id": "CartPole-v1", "seed": 2,
         "hyperparameters": {}}
    )
    counts, edges = np.histogram(np.random.default_rng(0).lognormal(0, 0.4, 500),
                                 bins=20, range=(0.0, 10.0))
    store.insert_gradient_stats(
        [
            _row(
                experiment_id,
                0,
                off_policy=True,
                is_weight_mean=1.02,
                is_weight_max=3.4,
                is_weight_p95=1.9,
                is_weight_ess=0.87,
                is_weight_histogram={"edges": [float(e) for e in edges],
                                     "counts": [int(c) for c in counts]},
            )
        ]
    )
    row = GradientStatRow(**store.gradient_stats_for(experiment_id)[0])
    assert row.off_policy is True
    assert row.is_weight_histogram is not None
    assert len(row.is_weight_histogram["counts"]) + 1 == len(row.is_weight_histogram["edges"]), (
        "numpy.histogram returns n+1 edges for n counts; a UI that zips them "
        "expects that and will silently drop the last bin otherwise"
    )
    assert 0.0 <= row.is_weight_ess <= 1.0, "ESS is stored as a fraction of n"


def test_on_policy_rows_leave_the_weight_columns_null():
    """Null, not zero.

    Zeros would average into `avg(is_weight_ess)` and make the on-policy arms
    look like catastrophically degenerate off-policy ones. This is the kind of
    thing that only bites when someone else queries your data — which is exactly
    the situation the data tier exists for.
    """
    store = get_store()
    experiment_id = store.insert_experiment(
        {"algorithm": "vpg", "env_id": "CartPole-v1", "seed": 3, "hyperparameters": {}}
    )
    store.insert_gradient_stats([_row(experiment_id, 0)])
    row = GradientStatRow(**store.gradient_stats_for(experiment_id)[0])
    assert row.is_weight_mean is None and row.is_weight_ess is None


def test_gradient_stats_endpoint_flags_degradation(client):
    store = get_store()
    experiment_id = store.insert_experiment(
        {"algorithm": "vpg+baseline", "env_id": "CartPole-v1", "seed": 9, "hyperparameters": {}}
    )
    store.insert_gradient_stats([_row(experiment_id, i) for i in range(4)])

    body = client.get(f"/gradient_stats?experiment_id={experiment_id}").json()
    assert body["count"] == 4
    assert "degraded" in body, (
        "the UI must be able to tell 'this run logged nothing' from 'the database "
        "is asleep'; an empty chart looks identical otherwise"
    )
    assert body["stats"][0]["policy_entropy"] == pytest.approx(0.69)


def test_gradient_stats_endpoint_rejects_a_missing_experiment_id(client):
    r = client.get("/gradient_stats")
    assert r.status_code == 422, "experiment_id is required; an unfiltered dump is not a default"
