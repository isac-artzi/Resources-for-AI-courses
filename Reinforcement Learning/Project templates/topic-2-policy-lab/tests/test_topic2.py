"""
The Topic 2 tests: the two agents, the environment they share, and the routing
that lets one service answer for both.

These sit alongside the standing four from the quality bar (which are in
tests/test_act_schema.py, tests/test_healthz.py, tests/test_store_roundtrip.py
and tests/test_no_torch.py and are not modified here). The rule about driving
the service through an HTTP test client still applies to everything that is a
service concern; the planner and the learner are library code and are tested
directly, because the thing under test is arithmetic and wrapping it in a
request would only obscure which line was wrong.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from envs import make_env
from shared.preprocess import dense_model
from train.compare import budget_statistics, rmse_against, t_ppf, t_sf
from train.monte_carlo import first_visit_mc_evaluation, mc_control_exploring_starts
from train.value_iteration import policy_evaluation, value_iteration

ROOT = pathlib.Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# The hand-solvable case.
# ---------------------------------------------------------------------------
#
# Four states in a 2x2 grid, two actions, gamma = 0.9. Small enough to solve on
# paper in about two minutes, which is the entire point: a planner tested only
# against a 25-state grid is a planner tested against its own output.
#
#     s0 --a0--> s1        s1 --a1--> s3   (+1, terminal)
#     s0 --a1--> s2        s1 --a0--> s1   (wall)
#                          s2 --a0--> s3   with probability 1/2, else stays
#                          s2 --a1--> s2   (wall)
#     s3 is terminal and absorbing.
#
# Working, from the leaves inward:
#
#     V(s3) = 0                                             (terminal)
#     V(s1) = max( 0.9 V(s1),  1 )                    = 1        -> a1
#     V(s2) = max( 0.5*1 + 0.5*0.9 V(s2),  0.9 V(s2) )
#           = 0.5 / (1 - 0.45)                        = 10/11    -> a0
#     V(s0) = max( 0.9 V(s1) = 0.9,  0.9 V(s2) = 0.818 ) = 0.9   -> a0
#
# Note there is no tie anywhere: the stochastic edge from s2 exists precisely
# so that the two routes out of s0 have different values. On a symmetric 2x2
# both actions from s0 are optimal, argmax picks the lower index, and the test
# passes for a reason that has nothing to do with the algorithm being correct.

HAND_P = {
    0: {0: [(1.0, 1, 0.0, False)], 1: [(1.0, 2, 0.0, False)]},
    1: {0: [(1.0, 1, 0.0, False)], 1: [(1.0, 3, 1.0, True)]},
    2: {0: [(0.5, 3, 1.0, True), (0.5, 2, 0.0, False)], 1: [(1.0, 2, 0.0, False)]},
    3: {0: [(1.0, 3, 0.0, True)], 1: [(1.0, 3, 0.0, True)]},
}
HAND_V = [0.9, 1.0, 10.0 / 11.0, 0.0]
HAND_POLICY = [0, 1, 0]  # non-terminal states only; s3's action is arbitrary


def test_value_iteration_recovers_the_hand_solvable_optimal_policy():
    plan = value_iteration(HAND_P, n_states=4, n_actions=2, gamma=0.9, theta=1e-14)
    assert plan.policy[:3].tolist() == HAND_POLICY
    assert plan.V.tolist() == pytest.approx(HAND_V, abs=1e-9)


def test_value_iteration_reports_an_error_bound_that_actually_bounds_the_error():
    """The contraction guarantee, checked rather than quoted: after stopping,
    ||V_k - V*||_inf must be under residual * gamma / (1 - gamma)."""
    plan = value_iteration(HAND_P, 4, 2, gamma=0.9, theta=1e-6)
    assert np.max(np.abs(plan.V - np.array(HAND_V))) <= plan.error_bound


def test_value_iteration_rejects_gamma_of_one():
    """gamma = 1 is legal for a guaranteed-terminating task but destroys the
    contraction argument, and with it the stopping rule and the error bound."""
    with pytest.raises(ValueError):
        value_iteration(HAND_P, 4, 2, gamma=1.0)


def test_the_planner_never_needs_the_environment_object():
    """value_iteration takes P, not an env. That is what makes the hand-solvable
    case above testable at all, and it is why Topic 5 can reuse this function."""
    plan = value_iteration(HAND_P, 4, 2, gamma=0.9)
    assert plan.sweeps > 0


# ---------------------------------------------------------------------------
# The environment.
# ---------------------------------------------------------------------------


def test_the_gridworld_is_a_gymnasium_environment_with_the_five_tuple_step():
    env = make_env()
    obs, info = env.reset(seed=0)
    assert isinstance(obs, int) and isinstance(info, dict)
    out = env.step(1)
    assert len(out) == 5, "code written against the old four-tuple Gym API will not run"
    _, reward, terminated, truncated, _ = out
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


def test_the_model_is_reachable_through_unwrapped_and_every_row_is_a_distribution():
    env = make_env()
    core = env.unwrapped
    for s in range(core.n_states):
        for a in range(core.n_actions):
            total = sum(p for p, _, _, _ in core.P[s][a])
            assert total == pytest.approx(1.0), f"P[{s}][{a}] sums to {total}"


def test_terminal_states_are_absorbing_with_zero_reward():
    """A terminal state left out of P crashes every planner; one given a
    non-zero self-loop reward makes the value function diverge quietly."""
    core = make_env().unwrapped
    for s in core.terminal_states:
        for a in range(core.n_actions):
            assert core.P[s][a] == [(1.0, s, 0.0, True)]


def test_exploring_starts_are_possible_through_reset_options():
    env = make_env()
    obs, _ = env.reset(seed=0, options={"state": 17})
    assert obs == 17


def test_the_time_limit_wrapper_truncates_rather_than_terminating():
    """A policy that never reaches the goal must produce a truncated episode,
    not an infinite loop — and `truncated` must be distinguishable from
    `terminated`, because only one of them means the return is complete."""
    env = make_env()
    env.reset(seed=0, options={"state": 0})
    truncated = False
    for _ in range(200):
        _, _, terminated, truncated, _ = env.step(0)  # up, into the top wall
        if terminated or truncated:
            break
    assert truncated and not terminated


def test_a_configurable_reward_function_changes_the_optimal_policy():
    """The reward function is configurable in the sense that matters: swapping
    it produces a different plan, not just a different number."""
    from envs import GridSpec, RewardSpec, make_gridworld

    baseline = make_gridworld(time_limit=False).unwrapped
    # Make the pits harmless and the step cost heavy: the agent should now be
    # willing to walk straight past a pit rather than around it.
    reckless = make_gridworld(
        GridSpec(reward=RewardSpec(step_cost=-0.5, pit_penalty=0.0)), time_limit=False
    ).unwrapped

    a = value_iteration(baseline.P, 25, 4).policy
    b = value_iteration(reckless.P, 25, 4).policy
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Planner and learner, cross-checked against each other.
# ---------------------------------------------------------------------------


def test_the_greedy_policys_exact_value_reproduces_the_planners_value_function():
    """Two routes to the same number: iterated Bellman backups, and a direct
    linear solve of (I - gamma P_pi) v = r_pi. If they disagree, one of them
    has the terminal bootstrap mask wrong."""
    core = make_env().unwrapped
    plan = value_iteration(core.P, core.n_states, core.n_actions)
    v_pi = policy_evaluation(core.P, plan.policy, core.n_states, core.n_actions)
    assert np.allclose(v_pi, plan.V, atol=1e-8)


def test_monte_carlo_evaluation_converges_toward_the_exact_solution():
    """The product's central claim, asserted coarsely on purpose.

    Monte Carlo error falls like 1/sqrt(n), so the RMSE at a larger budget is
    smaller IN EXPECTATION but not on every seed — a strict per-seed monotone
    assertion would be a flaky test that teaches the wrong lesson about
    stochastic estimators. Averaging three seeds and requiring a large drop
    between the ends, plus no increase larger than noise between neighbours, is
    the honest version of "it converges."
    """
    env = make_env()
    core = env.unwrapped
    interior = [s for s in range(core.n_states) if s not in core.terminal_states]
    plan = value_iteration(core.P, core.n_states, core.n_actions)

    budgets = (100, 1_000, 10_000)
    curves = []
    for seed in range(3):
        result = first_visit_mc_evaluation(
            env, plan.policy, episodes=budgets[-1], seed=seed,
            snapshot_at=budgets, collect_rows=False,
        )
        curves.append([rmse_against(result.snapshots[b], plan.V, interior) for b in budgets])

    mean = np.mean(curves, axis=0)
    assert mean[-1] < mean[0] / 4.0, f"RMSE barely moved across two decades: {mean}"
    for earlier, later in zip(mean, mean[1:]):
        assert later < earlier, f"mean RMSE increased with more episodes: {mean}"


def test_the_monte_carlo_estimate_lands_inside_the_planners_value_range():
    """A cheap sanity check that catches a discounting or off-by-one bug: a
    value estimate for the optimal policy cannot be wildly outside the exact
    value function's own range."""
    env = make_env()
    core = env.unwrapped
    interior = [s for s in range(core.n_states) if s not in core.terminal_states]
    plan = value_iteration(core.P, core.n_states, core.n_actions)
    est = first_visit_mc_evaluation(
        env, plan.policy, episodes=3_000, seed=0, collect_rows=False
    ).V
    lo, hi = plan.V[interior].min(), plan.V[interior].max()
    assert lo - 0.2 <= est[interior].min() and est[interior].max() <= hi + 0.2


def test_first_visit_and_every_visit_agree_to_within_sampling_noise():
    """Both estimators converge to v_pi. They differ in variance and in the
    independence of their samples, not in what they converge to."""
    env = make_env()
    core = env.unwrapped
    interior = [s for s in range(core.n_states) if s not in core.terminal_states]
    plan = value_iteration(core.P, core.n_states, core.n_actions)

    first = first_visit_mc_evaluation(env, plan.policy, episodes=5_000, seed=3,
                                      collect_rows=False).V
    every = first_visit_mc_evaluation(env, plan.policy, episodes=5_000, seed=3,
                                      every_visit=True, collect_rows=False).V
    assert rmse_against(first, every, interior) < 0.05


def test_monte_carlo_control_learns_a_policy_worth_almost_as_much_as_the_plan():
    """Control is scored on the VALUE of the policy it returns, not on how many
    cells match the planner's arrows. Several cells on this grid have near-tied
    actions, so arrow agreement is a noisy proxy for a quantity we can compute
    exactly."""
    env = make_env()
    core = env.unwrapped
    plan = value_iteration(core.P, core.n_states, core.n_actions)
    control = mc_control_exploring_starts(env, episodes=20_000, seed=0, collect_rows=False)
    learned = policy_evaluation(core.P, control.policy, core.n_states, core.n_actions)
    assert learned[core.start_state] > 0.9 * plan.V[core.start_state]


def test_every_logged_episode_carries_a_null_epsilon():
    """Exploring starts is not an epsilon-greedy schedule. Logging 0.0 would
    read as 'greedy with no exploration', which is the opposite of the truth."""
    env = make_env()
    core = env.unwrapped
    plan = value_iteration(core.P, core.n_states, core.n_actions)
    rows = first_visit_mc_evaluation(env, plan.policy, episodes=20, seed=0).episode_rows
    assert len(rows) == 20
    assert all(r["epsilon"] is None for r in rows)
    assert all(r["length"] >= 1 for r in rows)


def test_value_iteration_sweeps_are_logged_with_a_falling_residual():
    """One row per sweep, and the residual has to be monotone for a contraction
    — if it is not, the backup is not the one the error bound assumes."""
    core = make_env().unwrapped
    seen = []
    value_iteration(core.P, core.n_states, core.n_actions,
                    on_sweep=lambda k, V, r: seen.append((k, r)))
    assert [k for k, _ in seen] == list(range(len(seen)))
    residuals = [r for _, r in seen]
    assert all(b <= a + 1e-15 for a, b in zip(residuals, residuals[1:]))


# ---------------------------------------------------------------------------
# The statistics.
# ---------------------------------------------------------------------------


def test_the_self_contained_t_distribution_matches_scipy():
    """The fallback exists because SciPy is a training-tier dependency and CI
    installs the serving requirements only. A hand-rolled statistical function
    nobody validated is worse than no statistics at all, so validate it — and
    skip rather than pretend when SciPy is genuinely absent."""
    scipy_stats = pytest.importorskip("scipy.stats")
    import train.compare as compare

    saved = compare._scipy_stats
    compare._scipy_stats = None                      # force the fallback path
    try:
        for df in (1, 2, 9, 30, 200):
            for t in (-4.0, -1.0, 0.5, 1.833, 3.0):
                assert compare.t_sf(t, df) == pytest.approx(
                    float(scipy_stats.t.sf(t, df)), rel=1e-10, abs=1e-14
                )
            for q in (0.01, 0.05, 0.95, 0.975, 0.999):
                assert compare.t_ppf(q, df) == pytest.approx(
                    float(scipy_stats.t.ppf(q, df)), rel=1e-9, abs=1e-6
                )
    finally:
        compare._scipy_stats = saved


def test_the_t_quantile_is_the_one_the_report_quotes():
    """t_{0.95, 9} = 1.833, not 1.645. With ten seeds the normal approximation
    understates the interval by about 11%, and that gap is exactly the kind of
    thing a reader is entitled to assume you did not do."""
    assert t_ppf(0.95, 9) == pytest.approx(1.8331, abs=1e-4)
    assert t_ppf(0.975, 9) == pytest.approx(2.2622, abs=1e-4)
    assert t_sf(0.0, 9) == pytest.approx(0.5, abs=1e-12)


def test_the_equivalence_test_rejects_when_the_estimate_is_comfortably_inside_delta():
    stats = budget_statistics(10_000, [0.004, 0.005, 0.0045, 0.0052, 0.0038,
                                       0.0041, 0.0049, 0.0047, 0.0043, 0.0046], delta=0.0133)
    assert stats.equivalent_at_5pct
    assert stats.upper_95_bound < 0.0133
    assert stats.ci95_low < stats.mean_rmse < stats.ci95_high


def test_the_equivalence_test_does_not_reject_on_a_wide_estimate_inside_delta():
    """A mean below delta is NOT the claim. With enough spread the upper bound
    still sits outside the tolerance, and reporting a point estimate alone is
    exactly the failure this test guards against."""
    stats = budget_statistics(1_000, [0.001, 0.020, 0.002, 0.019, 0.003,
                                      0.018, 0.004, 0.017, 0.005, 0.016], delta=0.0133)
    assert stats.mean_rmse < 0.0133
    assert not stats.equivalent_at_5pct


def test_budget_statistics_refuses_a_single_seed():
    with pytest.raises(ValueError):
        budget_statistics(100, [0.01], delta=0.05)


# ---------------------------------------------------------------------------
# The service: two policy sources, one contract.
# ---------------------------------------------------------------------------


def test_act_routes_both_policy_sources_to_different_artifacts(client):
    seen = {}
    for source in ("value_iteration", "monte_carlo"):
        r = client.post("/act", json={"state": [0], "policy_source": source})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["policy_source"] == source
        assert body["policy_name"] == source, "the response must echo the RESOLVED artifact"
        assert isinstance(body["action"], int) and 0 <= body["action"] < 4
        seen[source] = body["policy_sha256"]

    assert seen["value_iteration"] != seen["monte_carlo"], (
        "both sources resolved to the same artifact — the mapping is not routing anything"
    )


def test_act_rejects_an_unknown_policy_source_with_a_422_listing_the_legal_values(client):
    r = client.post("/act", json={"state": [0], "policy_source": "linear_programming"})
    assert r.status_code == 422
    detail = str(r.json()["detail"])
    assert "value_iteration" in detail and "monte_carlo" in detail


def test_policy_source_overrides_policy_name(client):
    """Documented precedence, asserted. A field whose effect depends on which
    other field happens to be set is a field nobody can use safely."""
    r = client.post(
        "/act",
        json={"state": [0], "policy_name": "smoke_test_policy",
              "policy_source": "value_iteration"},
    )
    assert r.status_code == 200
    assert r.json()["policy_name"] == "value_iteration"


def test_an_unqualified_request_gets_the_exact_solution(client):
    """Three artifacts are registered, so 'default' cannot resolve by being the
    only one. It is configured, and it points at the plan: a caller who does
    not choose gets the answer that is provably right."""
    r = client.post("/act", json={"state": [0]})
    assert r.status_code == 200, r.text
    assert r.json()["policy_sha256"] == (
        client.post("/act", json={"state": [0], "policy_source": "value_iteration"})
        .json()["policy_sha256"]
    )


def test_rollout_accepts_a_policy_source_and_returns_bounded_evidence(client):
    r = client.post(
        "/rollout",
        json={"policy_source": "value_iteration", "episodes": 20, "seed": 0},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["policy_name"] == "value_iteration"
    assert len(body["returns"]) == 20
    assert body["stderr_return"] >= 0.0
    assert body["trajectory"], "the UI needs one trajectory to animate"


def test_rollout_is_hard_capped(client):
    """An unbounded rollout endpoint is how a free-tier instance is killed by
    its own users."""
    assert client.post("/rollout", json={"episodes": 100_000}).status_code == 422


def test_the_planner_outperforms_the_learner_on_a_shared_seed(client):
    """The comparison the client actually asked for, made through the API on
    one seed so that both agents face the same environment stream. One seed is
    NOT evidence — the seeded numbers in the README come from the ten-seed
    study — but it is enough to assert that the two agents are distinguishable
    through the service at all."""
    out = {}
    for source in ("value_iteration", "monte_carlo"):
        body = client.post(
            "/rollout", json={"policy_source": source, "episodes": 100, "seed": 7}
        ).json()
        out[source] = body["mean_return"]
    assert out["value_iteration"] >= out["monte_carlo"] - 3 * 0.05


def test_value_map_returns_both_grids_and_their_difference(client):
    body = client.get("/value_map").json()
    labels = {g["label"] for g in body["grids"]}
    assert {"value_iteration", "monte_carlo"} <= labels
    assert body["missing"] == []
    for grid in body["grids"]:
        assert len(grid["values"]) == grid["rows"] * grid["cols"] == 25
    diff = body["difference"]
    assert diff is not None and len(diff["values"]) == 25
    # The difference must actually be the difference, with the documented sign.
    by_label = {g["label"]: g["values"] for g in body["grids"]}
    for i in range(25):
        assert diff["values"][i] == pytest.approx(
            by_label["monte_carlo"][i] - by_label["value_iteration"][i], abs=1e-6
        )


def test_convergence_endpoint_names_its_source_and_flags_the_fallback(client):
    body = client.get("/convergence").json()
    assert body["source"] in ("data-tier", "checked-in-report", "none")
    if body["source"] == "checked-in-report":
        assert body["degraded"] is True, (
            "a chart served from a checked-in file must say so, or a stale "
            "result gets presented as a current one"
        )
    for point in body["points"]:
        assert point["ci95_low"] <= point["mean_rmse"] <= point["ci95_high"]


def test_the_serving_path_imports_neither_torch_nor_scipy():
    """The standing no-torch guard, extended to the dependency THIS topic added.

    Run in a subprocess, and that is the whole point. By the time this test
    executes, the pytest process has already imported `train.compare`, which
    imports SciPy — so an in-process assertion would fail for a reason that has
    nothing to do with what the deployed app loads. The deployed app imports
    `api.main` and nothing else, so that is what has to be measured, in a
    process that has imported nothing else. The base guard in
    tests/test_no_torch.py can assert in-process because nothing in the test
    suite imports torch at all; SciPy is a genuinely different case.
    """
    import subprocess
    import sys

    probe = (
        "import sys, api.main;"
        "assert 'torch' not in sys.modules, 'the app imported PyTorch (~490 MB)';"
        "assert 'scipy' not in sys.modules, 'the app imported SciPy (~90 MB, "
        "not in requirements-serve.txt — the deployed app would fail to start)';"
        "assert 'gymnasium' in sys.modules, 'the service runs bounded rollouts "
        "and does need the environment';"
        "print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], cwd=str(ROOT), capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_dense_model_and_the_planner_agree_on_the_hand_solvable_case():
    """Ties the preprocessing step to the algorithm that consumes it: one
    Bellman backup, written out by hand from the dense arrays, must reproduce
    the planner's first sweep."""
    T, R, B = dense_model(HAND_P, 4, 2)
    V0 = np.zeros(4)
    Q1 = R + 0.9 * (T * B) @ V0
    assert Q1.max(axis=1).tolist() == pytest.approx([0.0, 1.0, 0.5, 0.0])
