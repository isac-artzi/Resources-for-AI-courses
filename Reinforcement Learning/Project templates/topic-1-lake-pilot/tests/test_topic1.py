"""
Topic 1 tests — the ones that are specific to Lake Pilot. The four standing
tests required of every product live in the other files in this directory and
are not weakened here.

What these are for, in one line each:

  * The environment really is the slippery 8x8 lake — a flag that silently did
    nothing would make every number in the report a number about a different
    problem.
  * The epsilon schedule does what its string says.
  * Q-learning LEARNS. Asserted on the deterministic 4x4 lake, because that
    variant is fast, has an unambiguous right answer (a success rate of 1.0),
    and therefore gives a threshold a correct implementation clears every time.
    A test against the slippery 8x8 lake would need tens of thousands of
    episodes to reach a score whose spread across seeds is wide enough that any
    threshold is either flaky or vacuous.
  * The exported artifact means the same thing after a round trip through disk
    and `PolicyArtifactStore` as it did in memory.
  * /rollout works against the real environment, over HTTP.
"""

from __future__ import annotations

import numpy as np
import pytest

from api.policy import PolicyArtifactStore
from envs import ACTION_NAMES, action_space_size, make_env, observation_space_size
from shared.store import get_store
from train.export import export_qtable
from train.qlearning import evaluate_greedy, parse_eps_schedule, q_learning
from train.random_agent import run_random_agent

# The budget that makes the learning test reliable rather than lucky. Measured,
# not guessed: at 1,000 episodes only some seeds converge, at 2,000 every seed
# tried reached a success rate of 1.0 on the deterministic lake in about 0.2 s.
DETERMINISTIC_BUDGET = 2_000


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------


def test_make_env_is_the_slippery_8x8_lake():
    env = make_env()
    assert env.observation_space.n == 64 == observation_space_size()
    assert env.action_space.n == 4 == action_space_size() == len(ACTION_NAMES)

    # `is_slippery=True` is asserted against the transition kernel rather than
    # against the constructor argument, because the failure this catches is a
    # keyword that was accepted and ignored. Every non-terminal (state, action)
    # must have three possible outcomes at 1/3 each; with slip off there is one
    # outcome at probability 1.
    probs = [p for p, *_ in env.unwrapped.P[0][2]]
    assert len(probs) == 3, f"expected a stochastic kernel, got {probs}"
    assert all(abs(p - 1 / 3) < 1e-9 for p in probs)
    env.close()


def test_the_deterministic_variant_really_is_deterministic():
    env = make_env("4x4", is_slippery=False)
    assert [p for p, *_ in env.unwrapped.P[0][2]] == [1.0]
    env.close()


# ---------------------------------------------------------------------------
# The epsilon schedule
# ---------------------------------------------------------------------------


def test_linear_schedule_decays_then_holds():
    f = parse_eps_schedule("linear:1.0:0.05:0.5")
    assert f(0, 1000) == pytest.approx(1.0)
    assert f(250, 1000) == pytest.approx(0.525, abs=1e-6)
    # Past the decay fraction it must be FLAT, not still falling: the flat tail
    # is the only stretch of training whose return is comparable to the greedy
    # score.
    assert f(500, 1000) == pytest.approx(0.05)
    assert f(999, 1000) == pytest.approx(0.05)


def test_exponential_schedule_is_floored_and_constant_schedule_is_constant():
    f = parse_eps_schedule("exp:1.0:0.1:0.99")
    assert f(0, 1000) == pytest.approx(1.0)
    assert f(10_000, 20_000) == pytest.approx(0.1), "the floor must hold, not underflow"
    g = parse_eps_schedule("const:0.3")
    assert g(0, 100) == g(99, 100) == 0.3


def test_a_malformed_schedule_fails_loudly_at_parse_time():
    # Loudly and EARLY: a schedule typo that fell through to a default would
    # cost the length of a training run before anyone noticed.
    for bad in ("linear:1.0:0.05", "quadratic:1:0:0.5", "linear:1.0:0.05:0"):
        with pytest.raises(ValueError):
            parse_eps_schedule(bad)


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------


def test_qlearning_recovers_a_greedy_policy_on_the_deterministic_lake():
    """The load-bearing test: after training, the greedy policy actually works.

    On the deterministic 4x4 lake the optimal success rate is 1.0, so >0.9 is a
    real threshold rather than a hedge. alpha is high (0.8) precisely because
    the environment is deterministic: with no transition noise the target is
    exact, and a small alpha would only slow convergence down.
    """
    result = q_learning(
        episodes=DETERMINISTIC_BUDGET,
        seed=0,
        alpha=0.8,
        gamma=0.95,
        eps_schedule="linear:1.0:0.05:0.5",
        eval_every=0,
        map_name="4x4",
        is_slippery=False,
        log=False,
        progress=False,
    )
    assert result.Q.shape == (16, 4)

    ev = evaluate_greedy(result.Q, episodes=100, map_name="4x4", is_slippery=False)
    assert ev["mean_return"] > 0.9, (
        f"greedy success rate {ev['mean_return']:.2f} on the DETERMINISTIC lake. "
        "This variant has an unambiguous optimum of 1.0, so a score this low is "
        "an implementation problem, not variance. Check the update line, and "
        "check that terminated (not truncated) is what zeroes the bootstrap."
    )


def test_qlearning_logs_every_episode_with_the_epsilon_in_force():
    store = get_store()
    episodes = 300
    result = q_learning(
        episodes=episodes,
        seed=1,
        eps_schedule="linear:1.0:0.1:0.5",
        eval_every=0,
        map_name="4x4",
        is_slippery=True,
        log=True,
        progress=False,
    )
    rows = store.episodes_for(result.experiment_id)
    assert len(rows) == episodes, "EVERY episode, not a sample: the curve is the deliverable"
    assert [r["episode_index"] for r in rows] == list(range(episodes))
    assert all(r["epsilon"] is not None for r in rows)
    # Non-increasing, and it must actually have moved. A schedule wired up but
    # never called would still satisfy "not None" on every row.
    eps = [r["epsilon"] for r in rows]
    assert all(a >= b - 1e-12 for a, b in zip(eps, eps[1:]))
    assert eps[0] > eps[-1]


def test_periodic_greedy_evaluations_are_recorded_separately_from_episodes():
    store = get_store()
    result = q_learning(
        episodes=200,
        seed=2,
        eval_every=100,
        eval_episodes=20,
        map_name="4x4",
        is_slippery=False,
        log=True,
        progress=False,
    )
    assert len(result.evaluations) == 2
    written = [
        e for e in store.evaluations if e["experiment_id"] == result.experiment_id
    ]
    assert [e["at_training_episode"] for e in written] == [100, 200]
    assert all("stderr_return" in e for e in written), (
        "an evaluation row without a standard error is a mean with no idea how "
        "well it is known"
    )


def test_random_agent_writes_one_row_per_episode_and_reports_a_standard_error():
    store = get_store()
    r = run_random_agent(episodes=50, seed=0, map_name="4x4", is_slippery=True, log=True)
    rows = store.episodes_for(r["experiment_id"])
    assert len(rows) == 50
    # epsilon = 1.0, because a uniform policy IS epsilon-greedy with epsilon
    # pinned at one. Logging it that way puts the baseline and the learner on
    # one axis instead of leaving them as two unrelated things.
    assert all(row["epsilon"] == 1.0 for row in rows)
    assert r["stderr_return"] >= 0.0
    assert 0.0 <= r["mean_return"] <= 1.0


# ---------------------------------------------------------------------------
# The artifact seam
# ---------------------------------------------------------------------------


def test_exported_artifact_roundtrips_and_argmaxes_identically(tmp_path):
    """The seam between the tiers, checked in the only way that matters.

    Equality of the ARGMAX for every state, not equality of the array. The
    export deliberately narrows to float32 to halve the artifact, so an exact
    array comparison would fail for a reason that has no effect on behaviour.
    What must survive the round trip is the decision.
    """
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(64, 4))

    row = export_qtable(Q, tmp_path / "roundtrip.npz")
    assert row["bytes"] > 0 and len(row["sha256"]) == 64
    assert row["kind"] == "tabular" and row["obs_dim"] == 1 and row["n_actions"] == 4

    store = PolicyArtifactStore(tmp_path)
    policy, meta = store.get("roundtrip")
    assert meta["sha256"] == row["sha256"], "the registry row must describe the file on disk"

    for state in range(64):
        served, value = policy.act(np.asarray([state], dtype=np.float64), deterministic=True)
        assert served == int(np.argmax(Q[state])), f"state {state} argmax changed on export"
        assert value == pytest.approx(float(np.max(Q[state])), abs=1e-5)


def test_an_exported_artifact_contains_no_pickled_objects(tmp_path):
    # allow_pickle=False is what the service tier uses. A .npz that only loads
    # with pickling enabled is arbitrary code execution wearing an array's
    # clothes, and it would fail in production rather than here.
    export_qtable(np.zeros((16, 4)), tmp_path / "plain.npz")
    with np.load(tmp_path / "plain.npz", allow_pickle=False) as z:
        assert set(z.files) == {"Q"}


# ---------------------------------------------------------------------------
# The service tier, over HTTP, against the real environment
# ---------------------------------------------------------------------------


def test_rollout_runs_real_episodes_against_the_real_environment(client):
    r = client.post(
        "/rollout",
        json={
            "policy_name": "lake_smoke_policy",
            "episodes": 3,
            "max_steps": 60,
            "seed": 0,
            "record_trajectory": True,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["returns"]) == 3 == body["episodes"]
    assert body["stderr_return"] >= 0.0
    assert body["trajectory"], "the UI cannot animate what the service did not return"

    for t in body["trajectory"]:
        # Every state is a one-element vector holding an index into the 64-square
        # lake. If this ever fails, /act's dimensionality check and the artifact's
        # obs_dim have drifted apart from the environment.
        assert len(t["state"]) == 1
        assert 0 <= int(t["state"][0]) < 64
        assert 0 <= t["action"] < 4
    assert body["trajectory"][0]["step"] == 0


def test_rollout_is_reproducible_under_a_named_seed(client):
    payload = {"policy_name": "lake_smoke_policy", "episodes": 3, "max_steps": 60, "seed": 7}
    first = client.post("/rollout", json=payload).json()
    second = client.post("/rollout", json=payload).json()
    # The greedy path must be reproducible from the seed alone, or "seed 7" in
    # the README names nothing. (The sampled path is not — see the note on
    # RolloutRequest.deterministic.)
    assert first["returns"] == second["returns"]


def test_the_sampled_rollout_path_is_genuinely_random(client):
    """The 'untrained' side of the Watch tab, checked end to end.

    With an all-zero Q-table every action ties, so the softmax is uniform and
    the sampled path must produce more than one distinct action. The greedy
    path on the same table always returns action 0 — that difference IS the
    random-versus-trained toggle, and it is worth seeing asserted.
    """
    body = client.post(
        "/rollout",
        json={
            "policy_name": "lake_smoke_policy",
            "episodes": 2,
            "max_steps": 80,
            "seed": 0,
            "deterministic": False,
        },
    ).json()
    actions = {t["action"] for t in body["trajectory"]}
    assert len(actions) > 1, f"a uniform policy produced only {actions}"

    greedy = client.post(
        "/rollout",
        json={"policy_name": "lake_smoke_policy", "episodes": 1, "max_steps": 20, "seed": 0},
    ).json()
    assert {t["action"] for t in greedy["trajectory"]} == {0}


def test_episodes_endpoint_returns_a_curve_the_compare_tab_can_plot(client):
    r = run_random_agent(episodes=25, seed=3, map_name="4x4", is_slippery=True, log=True)
    body = client.get(f"/episodes?experiment_id={r['experiment_id']}").json()
    assert body["count"] == 25 and body["degraded"] is False
    point = body["points"][0]
    # Serialised under the SQL column name, not under the Python attribute
    # name: `return` is a keyword, and a UI reading `return_` would silently
    # plot nothing.
    assert "return" in point and "epsilon" in point
    assert body["truncated"] is False


def test_episodes_endpoint_says_when_it_truncated(client):
    r = run_random_agent(episodes=30, seed=4, map_name="4x4", is_slippery=True, log=True)
    body = client.get(f"/episodes?experiment_id={r['experiment_id']}&limit=10").json()
    assert body["count"] == 10
    assert body["truncated"] is True, (
        "a curve that stops early without saying so looks like an agent that "
        "stopped learning"
    )
