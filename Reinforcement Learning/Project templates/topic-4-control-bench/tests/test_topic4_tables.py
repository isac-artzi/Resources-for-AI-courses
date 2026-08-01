"""
The round trip for the two tables 002_topic4.sql adds: write rows, read them
back, and check that the shapes the Bake-Off and Entropy Sweep tabs depend on
survive the journey.

Against a configured project this exercises Supabase. In CI, with no
credentials, it exercises the in-process fallback — the SAME interface and the
same row shapes, so a column you forgot to add to `shared/schemas.py` still
shows up here as a validation error. What it cannot catch is a Postgres-side
error (the unique constraints, the foreign keys, the `check` on `mode`); apply
the migration and run this against your real project at least once before
submitting.
"""

from __future__ import annotations

import pytest

from shared.schemas import EntropySweepRow, PolicyUpdateRow
from shared.store import get_store


def _update_row(experiment_id: str, i: int, **over) -> dict:
    return {
        "experiment_id": experiment_id,
        "update_index": i,
        "env_steps": (i + 1) * 1024,
        "episode_index": (i + 1) * 2,
        "policy_loss": -0.1 * i,
        "value_loss": 10.0 / (i + 1),
        "policy_entropy": 1.09 - 0.01 * i,
        "kl_divergence": 0.004 + 0.0001 * i,
        "clip_fraction": 0.02,
        "alpha": None,
    } | over


def test_policy_updates_write_read_roundtrip():
    store = get_store()
    experiment_id = store.insert_experiment(
        {"algorithm": "ppo", "env_id": "Acrobot-v1", "seed": 0, "hyperparameters": {"clip": 0.2}}
    )
    rows = [_update_row(experiment_id, i) for i in range(8)]
    assert store.insert_policy_updates(rows) == 8

    read_back = store.policy_updates_for(experiment_id)
    assert len(read_back) == 8
    assert [r["update_index"] for r in read_back] == list(range(8)), (
        "rows must come back in update order — the chart plots them as a line and "
        "an unordered read draws a scribble"
    )
    assert read_back[0]["env_steps"] == 1024, (
        "env_steps is what lets the KL series and the learning curve share an "
        "x-axis with a PPO run whose episode lengths change during training"
    )


def test_a_null_kl_stays_null():
    """A2C and SAC have no KL, and that must survive the round trip as NULL.

    The failure this guards against is subtle and expensive: if a null became
    0.0 anywhere in the pipeline, `avg(kl_divergence)` over all algorithms would
    include those zeros and A2C would appear to be the most conservative method
    in the study — a confident, wrong, publishable-looking finding.
    """
    store = get_store()
    experiment_id = store.insert_experiment(
        {"algorithm": "a2c", "env_id": "CartPole-v1", "seed": 1, "hyperparameters": {}}
    )
    store.insert_policy_updates(
        [_update_row(experiment_id, i, kl_divergence=None, clip_fraction=None) for i in range(3)]
    )
    read_back = store.policy_updates_for(experiment_id)
    assert all(r["kl_divergence"] is None for r in read_back)
    # And the model must accept the null rather than coercing it.
    parsed = [PolicyUpdateRow(**r) for r in read_back]
    assert all(p.kl_divergence is None for p in parsed)


def test_every_update_row_validates_against_the_schema():
    """The row dict and the Pydantic model must agree, or the endpoint 500s.

    This is the cheap version of the migration check: shared/schemas.py and
    db/migrations/002_topic4.sql are supposed to describe the same row, and
    nothing enforces that except a test that builds one from the other.
    """
    store = get_store()
    experiment_id = store.insert_experiment(
        {"algorithm": "sac", "env_id": "Pendulum-v1", "seed": 2, "hyperparameters": {}}
    )
    store.insert_policy_updates(
        [_update_row(experiment_id, i, kl_divergence=None, alpha=0.2) for i in range(4)]
    )
    for r in store.policy_updates_for(experiment_id):
        PolicyUpdateRow(**r)


def _sweep_row(experiment_id: str, label: str, seed: int, **over) -> dict:
    return {
        "experiment_id": experiment_id,
        "mode": "auto" if label == "auto" else "fixed",
        "alpha_setting": label,
        "alpha_value": 0.2,
        "seed": seed,
        "episodes": 75,
        "env_steps": 15_000,
        "episodes_to_threshold": 40,
        "threshold": -300.0,
        "mean_return_last_100": -180.0,
        "return_std_last_100": 90.0,
        "mean_policy_entropy": -0.9,
        "eval_mean_return": -150.0,
    } | over


def test_entropy_sweep_roundtrip_and_null_convergence():
    """A run that never reached the threshold stores NULL, not a sentinel.

    A large sentinel — 9999, say — would be averaged into
    `avg(episodes_to_threshold)` and would invent a slow convergence where there
    was none at all. The summary view reports the mean over the runs that
    reached the bar AND how many did, precisely so the reader can see both.
    """
    store = get_store()
    written = []
    for label in ("alpha=0.5", "alpha=0.01", "auto"):
        for seed in range(3):
            eid = store.insert_experiment(
                {"algorithm": "sac", "env_id": "Pendulum-v1", "seed": seed,
                 "hyperparameters": {"alpha_setting": label}}
            )
            row = _sweep_row(
                eid, label, seed,
                episodes_to_threshold=None if label == "alpha=0.5" else 40,
            )
            store.insert_entropy_sweep(row)
            written.append(row)

    rows = store.entropy_sweep_rows(200)
    mine = {(r["alpha_setting"], r["seed"]) for r in rows}
    for row in written:
        assert (row["alpha_setting"], row["seed"]) in mine

    parsed = [EntropySweepRow(**r) for r in rows if (r["alpha_setting"], r["seed"]) in mine]
    assert len(parsed) >= 9, "three arms times three seeds is the minimum the brief asks for"
    never = [p for p in parsed if p.alpha_setting == "alpha=0.5"]
    assert never and all(p.episodes_to_threshold is None for p in never)


def test_the_mode_field_is_constrained():
    """`mode` is one of two words, and the schema says so.

    The migration has a `check` constraint; Pydantic has a Literal. Both, because
    the fallback store has no constraints at all and CI runs against the
    fallback — without the Literal, a typo would pass every test here and fail
    only against Postgres, which is the one place you cannot easily debug it.
    """
    with pytest.raises(Exception):
        EntropySweepRow(**_sweep_row("exp-x", "auto", 0, mode="automatic"))


def test_the_sweep_summary_matches_the_python_aggregate():
    """`train.entropy_sweep.summarise` and the SQL view must define the same numbers.

    Only the Python half is executed here — the view needs Postgres. What this
    catches is the half that is easy to get wrong silently: averaging
    `episodes_to_threshold` over all runs instead of over the ones that reached
    the bar. The view uses `count(col)`, which skips nulls, and this asserts the
    Python does the same.
    """
    from train.entropy_sweep import summarise

    rows = [
        _sweep_row("e1", "alpha=0.5", 0, mean_return_last_100=-900.0,
                   episodes_to_threshold=None),
        _sweep_row("e2", "alpha=0.5", 1, mean_return_last_100=-700.0,
                   episodes_to_threshold=60),
        _sweep_row("e3", "alpha=0.5", 2, mean_return_last_100=-800.0,
                   episodes_to_threshold=None),
    ]
    out = summarise(rows)
    arm = next(a for a in out if a["alpha_setting"] == "alpha=0.5")
    assert arm["seeds"] == 3
    assert arm["reached_threshold"] == 1
    assert arm["mean_episodes_to_threshold"] == pytest.approx(60.0), (
        "the mean must be over the runs that REACHED the threshold; averaging in "
        "the two that did not requires inventing a value for them"
    )
    assert arm["mean_final_return"] == pytest.approx(-800.0)
