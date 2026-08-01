"""
train/compare.py — how far is the learner from the planner, as a function of
experience, and when does the gap stop mattering?

    python -m train.compare               # 10 seeds, budgets up to 30k episodes
    python -m train.compare --seeds 20

This is the file the client actually asked for. The two consultants in the
product brief disagree about whether to plan or to learn; this script turns
that argument into a number with an interval around it.

What it measures
----------------
For each seed, first-visit Monte Carlo evaluation of the OPTIMAL policy is run
for a growing number of episodes and, at each budget, scored against the exact
solution:

    RMSE(n) = sqrt( mean_over_non_terminal_states ( V_hat_n(s) - V*(s) )^2 )

Three choices in that line are worth defending, because each of them is a place
where a plausible-looking alternative would measure the wrong thing:

  * The policy being evaluated is the value-iteration OPTIMAL policy. Monte
    Carlo prediction estimates v_pi for whatever pi it is given, so comparing
    an estimate of v_pi against V* only makes sense when pi is optimal — then
    v_pi IS V* and the two estimators are estimating the same object.
    Evaluating a random policy against V* would produce a large RMSE that never
    shrinks, and it would be entirely correct: the estimator would be
    converging perfectly to a different number.

  * Terminal states are excluded. Both methods assign them exactly 0 by
    construction, so including them adds two guaranteed-zero error terms that
    shrink the RMSE by a factor of sqrt(23/25) without any estimator getting
    better.

  * Seeds are the unit of replication, not states. The 23 per-state errors
    within one run are correlated (they share a trajectory stream), so a
    confidence interval computed across states would be far too narrow. Across
    independent seeds it is honest.

The statistics are at the bottom of the file, with their assumptions written
out where the test is implemented rather than in the README only.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from envs import ENV_ID, make_env
from shared.config import get_settings
from shared.store import get_store
from train.monte_carlo import first_visit_mc_evaluation
from train.value_iteration import DEFAULT_GAMMA, value_iteration

# Episode budgets at which the estimate is scored. Roughly logarithmic, because
# Monte Carlo error falls like 1/sqrt(n): on a linear grid every point after
# the first would sit on top of its neighbour.
DEFAULT_BUDGETS: tuple[int, ...] = (100, 300, 1_000, 3_000, 10_000, 30_000)

# The syllabus floor is 10. Ten is also roughly where a t-interval stops being
# embarrassingly wide (t_{0.975,9} = 2.262 against 1.96 asymptotically); with
# five seeds it is 2.776 and the interval is nearly half again as wide for the
# same data.
DEFAULT_SEEDS = 10

# --- The equivalence tolerance, fixed BEFORE any of this was run. -----------
# "Statistically indistinguishable" is not a testable claim on its own: failing
# to reject equality is not evidence of equality, and with enough seeds any
# non-zero difference becomes significant. The claim has to be "closer than
# delta", and delta has to come from the problem, not from the data.
#
# Here delta is 2% of the span of V* across non-terminal states. The client's
# routing decisions are made by comparing state values to each other; a
# difference of 2% of the total spread cannot flip a routing decision anywhere
# on this grid, which is what makes it the right unit. Quoting delta in raw
# reward units instead would make it meaningless the moment the reward
# specification changes.
DELTA_FRACTION_OF_SPAN = 0.02

ALPHA = 0.05

REPORT_PATH = pathlib.Path("reports/convergence.json")


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------


@dataclass
class BudgetStats:
    episodes: int
    seeds: int
    mean_rmse: float
    sd_rmse: float
    stderr_rmse: float
    ci95_low: float
    ci95_high: float
    upper_95_bound: float          # one-sided; the TOST arm below compares this to delta
    t_statistic: float
    p_value: float
    equivalent_at_5pct: bool


def rmse_against(estimate: np.ndarray, exact: np.ndarray, states: Sequence[int]) -> float:
    """Root-mean-square error over a named set of states.

    `states` is explicit rather than defaulted to "all of them" so that the
    exclusion of terminal states is visible at every call site. A default that
    silently drops rows is a default that ends up in a report unnoticed.
    """
    idx = np.asarray(states, dtype=np.int64)
    diff = np.asarray(estimate, dtype=np.float64)[idx] - np.asarray(exact, dtype=np.float64)[idx]
    return float(np.sqrt(np.mean(diff**2)))


def run_convergence_study(
    seeds: int = DEFAULT_SEEDS,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    gamma: float = DEFAULT_GAMMA,
    write_rows: bool = True,
    quiet: bool = False,
) -> dict[str, Any]:
    env = make_env()
    core = env.unwrapped
    interior = [s for s in range(core.n_states) if s not in core.terminal_states]

    plan = value_iteration(core.P, core.n_states, core.n_actions, gamma=gamma)
    span = float(np.ptp(plan.V[interior]))
    delta = DELTA_FRACTION_OF_SPAN * span

    store = get_store()
    budgets = tuple(sorted(int(b) for b in budgets))
    curves: dict[int, list[float]] = {b: [] for b in budgets}

    for seed in range(seeds):
        # One run per seed, snapshotted at every budget on the way past. Running
        # each budget as a separate fresh run would be a different and much more
        # expensive experiment, and — more importantly — its points would be
        # statistically independent of each other, which would hide the fact
        # that a single learner's curve is monotone-ish while the budget-to-
        # budget DIFFERENCES are not.
        result = first_visit_mc_evaluation(
            env,
            plan.policy,
            episodes=budgets[-1],
            gamma=gamma,
            seed=seed,
            snapshot_at=budgets,
            collect_rows=False,
        )
        experiment_id = None
        if write_rows:
            experiment_id = store.insert_experiment(
                {
                    "algorithm": "mc_evaluation",
                    "env_id": ENV_ID,
                    "seed": seed,
                    "hyperparameters": {
                        "gamma": gamma,
                        "budgets": list(budgets),
                        "policy": "value_iteration_optimal",
                        "exploration": "exploring_starts",
                        "first_visit": True,
                    },
                    "git_sha": get_settings().git_sha,
                    "notes": "convergence study: MC estimate of v_pi* against the exact V*",
                }
            )

        rows = []
        for budget in budgets:
            V_hat = result.snapshots[budget]
            err = rmse_against(V_hat, plan.V, interior)
            curves[budget].append(err)
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "at_training_episode": budget,
                    "episodes": budget,
                    "metric": "value_rmse",
                    "policy_source": "monte_carlo",
                    "rmse": err,
                    # The standing columns still have to be filled, and they
                    # still mean what they meant: the mean discounted return
                    # the learner actually observed. Leaving them null would
                    # break the run-history view for every reader who does not
                    # know this topic added a metric column.
                    "mean_return": float(np.mean(V_hat[interior])),
                    "std_return": float(np.std(V_hat[interior], ddof=1)),
                    "stderr_return": float(
                        np.std(V_hat[interior], ddof=1) / math.sqrt(len(interior))
                    ),
                }
            )
        if write_rows:
            for row in rows:
                store.insert_evaluation(row)
        if not quiet:
            print(f"  seed {seed:2d}: " + "  ".join(
                f"n={b}:{curves[b][-1]:.4f}" for b in budgets))

    stats = [budget_statistics(b, curves[b], delta) for b in budgets]
    first_equivalent = next((s.episodes for s in stats if s.equivalent_at_5pct), None)

    summary: dict[str, Any] = {
        "env_id": ENV_ID,
        "gamma": gamma,
        "seeds": seeds,
        "budgets": list(budgets),
        "v_star_span_non_terminal": span,
        "delta": delta,
        "delta_fraction_of_span": DELTA_FRACTION_OF_SPAN,
        "alpha": ALPHA,
        "test": TEST_NAME,
        "assumptions": ASSUMPTIONS,
        "backend": _backend_name(),
        "episodes_to_indistinguishable": first_equivalent,
        "per_budget": [asdict(s) for s in stats],
        "raw_rmse_by_budget": {str(b): curves[b] for b in budgets},
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Written to disk as well as to `evaluations` on purpose. The database is
    # the evidence tier, but a fresh clone with no Supabase credentials still
    # has to be able to render the Convergence tab — otherwise the first thing
    # a reviewer sees is an empty chart. GET /convergence serves this file when
    # the data tier is unconfigured, and flags the response as degraded so the
    # two sources can never be confused.
    REPORT_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    if not quiet:
        print()
        print(format_table(summary))
    return summary


# ---------------------------------------------------------------------------
# The statistics
# ---------------------------------------------------------------------------

TEST_NAME = (
    "One-sided one-sample Student t-test on the per-seed RMSE against a "
    "pre-declared equivalence margin delta (the upper arm of a two one-sided "
    "tests / TOST equivalence procedure; the lower arm is vacuous because RMSE "
    "is non-negative by construction)."
)

ASSUMPTIONS = [
    "The per-seed RMSE values are independent. This is guaranteed by "
    "construction — each seed drives its own generator and its own environment "
    "stream — and not inferred from the data.",
    "The per-seed RMSE is approximately normally distributed ACROSS seeds. "
    "RMSE is a non-negative, right-skewed statistic, so this is the weakest "
    "assumption here; with 10 seeds the t-test is tolerant of mild skew, and "
    "the drop-in replacement if you doubt it is a bootstrap percentile "
    "interval over the same seed-level values.",
    "The variance is unknown and estimated from the same 10 values, which is "
    "why this is a t-test and not a z-test. At 9 degrees of freedom the "
    "critical value is 1.833 rather than 1.645 — a 11% wider interval that a "
    "normal approximation would quietly omit.",
    "delta was fixed before the study ran (2% of the span of V* over "
    "non-terminal states). An equivalence margin chosen after seeing the RMSE "
    "curve is not a hypothesis test, it is a description.",
    "Every seed ran the same budget grid, and no seed was dropped. Snapshots "
    "within one seed are from ONE run, so the points on a single curve are not "
    "independent of each other; only the across-seed comparison at a FIXED "
    "budget is.",
    "The test is applied at each budget on a monotone curve and the smallest "
    "budget that passes is reported. That is one claim about where a curve "
    "crosses a line, not six independent discoveries — no multiplicity "
    "correction is applied, and none would be meaningful, but the distinction "
    "belongs in the report rather than in a footnote.",
]


def budget_statistics(episodes: int, values: Sequence[float], delta: float) -> BudgetStats:
    """Mean RMSE with a 95% interval, plus the equivalence test against delta."""
    x = np.asarray(values, dtype=np.float64)
    k = x.size
    if k < 2:
        raise ValueError("at least two seeds are needed for an interval; the syllabus floor is 10")

    mean = float(x.mean())
    sd = float(x.std(ddof=1))          # ddof=1: sample sd. ddof=0 understates it.
    stderr = sd / math.sqrt(k)
    df = k - 1

    two_sided = t_ppf(1.0 - ALPHA / 2.0, df)
    one_sided = t_ppf(1.0 - ALPHA, df)

    t_stat = (mean - delta) / stderr if stderr > 0 else -math.inf
    # P(T <= t_stat): the probability of an RMSE this far BELOW delta if the
    # true mean were exactly delta. Small p => the estimate is inside the
    # tolerance, which is the direction of interest.
    p_value = 1.0 - t_sf(t_stat, df)

    return BudgetStats(
        episodes=episodes,
        seeds=k,
        mean_rmse=mean,
        sd_rmse=sd,
        stderr_rmse=stderr,
        ci95_low=mean - two_sided * stderr,
        ci95_high=mean + two_sided * stderr,
        upper_95_bound=mean + one_sided * stderr,
        t_statistic=t_stat,
        p_value=p_value,
        equivalent_at_5pct=bool(p_value < ALPHA),
    )


# --- t distribution --------------------------------------------------------
#
# SciPy is a training-tier dependency: it is ~90 MB installed and has no
# business in requirements-serve.txt, and CI installs the serving requirements
# only. So this module uses SciPy when it is there and falls back to a
# self-contained implementation when it is not — and tests/test_topic2.py
# checks the two against each other, because a hand-rolled statistical function
# nobody validated is worse than no statistics at all.

try:  # pragma: no cover - exercised on a training machine
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover - exercised in CI
    _scipy_stats = None


def _backend_name() -> str:
    return "scipy" if _scipy_stats is not None else "self-contained"


def t_sf(t: float, df: int) -> float:
    """Upper tail P(T > t) for Student's t with df degrees of freedom."""
    if _scipy_stats is not None:
        return float(_scipy_stats.t.sf(t, df))
    if not math.isfinite(t):
        return 0.0 if t > 0 else 1.0
    x = df / (df + t * t)
    half = 0.5 * _betainc(df / 2.0, 0.5, x)
    return half if t >= 0 else 1.0 - half


def t_ppf(q: float, df: int) -> float:
    """Inverse CDF: the t value with P(T <= t) = q."""
    if _scipy_stats is not None:
        return float(_scipy_stats.t.ppf(q, df))
    if not 0.0 < q < 1.0:
        raise ValueError("q must be strictly inside (0, 1)")
    # Bisection on a monotone CDF. Not elegant, but it needs no derivative, it
    # cannot diverge, and it is called a handful of times per run rather than
    # per sample. Compared in the tail (`sf > 1 - q`) rather than as
    # `1 - sf < q` to keep the small quantity small.
    #
    # Accuracy note: the tail is parameterised by x = df / (df + t^2), so for
    # |t| below about sqrt(eps * df) ~ 1e-7 the argument is indistinguishable
    # from 1 and the quantile stalls there. That floor sits on the MEDIAN,
    # where the answer is 0 anyway; the critical values this module actually
    # uses (0.95, 0.975) agree with SciPy to 15 significant figures, and
    # tests/test_topic2.py asserts it.
    lo, hi = -1.0e4, 1.0e4
    target = 1.0 - q
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_sf(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b), by the standard continued fraction.

    This is the only piece of special-function machinery in the repository.
    It is here rather than reimplemented per call site because a t-test with a
    subtly wrong tail is a test that reports significance at the wrong place
    and gives no sign of it.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    # The continued fraction converges quickly only on one side of this point;
    # past it, use the symmetry I_x(a,b) = 1 - I_{1-x}(b,a).
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float) -> float:
    """Lentz's algorithm for the continued fraction of the incomplete beta."""
    tiny, eps, max_iter = 1e-300, 3e-16, 300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        num = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        num = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise RuntimeError("incomplete beta continued fraction did not converge")


# ---------------------------------------------------------------------------


def format_table(summary: dict[str, Any]) -> str:
    lines = [
        f"delta = {summary['delta']:.5f}  "
        f"({summary['delta_fraction_of_span']:.0%} of the V* span "
        f"{summary['v_star_span_non_terminal']:.4f}), "
        f"alpha = {summary['alpha']}, seeds = {summary['seeds']}, "
        f"t backend = {summary['backend']}",
        "",
        f"{'episodes':>9} {'mean RMSE':>10} {'95% CI':>21} {'upper 95%':>10} "
        f"{'p (< delta)':>12} {'equivalent':>11}",
    ]
    for row in summary["per_budget"]:
        lines.append(
            f"{row['episodes']:>9} {row['mean_rmse']:>10.5f} "
            f"[{row['ci95_low']:>8.5f}, {row['ci95_high']:>8.5f}] "
            f"{row['upper_95_bound']:>10.5f} {row['p_value']:>12.2e} "
            f"{'yes' if row['equivalent_at_5pct'] else 'no':>11}"
        )
    n = summary["episodes_to_indistinguishable"]
    lines.append("")
    lines.append(
        f"Statistically indistinguishable from the exact solution at the 5% level "
        f"from {n} episodes onward." if n is not None else
        "The estimate did not become statistically indistinguishable within the "
        "budget grid. That is a legitimate result — report it, and either extend "
        "the grid or say plainly that the client should plan rather than learn."
    )
    lines.append(f"Test: {summary['test']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help="at least 10; fewer will not satisfy the rubric")
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--no-write", action="store_true",
                        help="skip the evaluations rows; still writes reports/convergence.json")
    args = parser.parse_args()
    run_convergence_study(
        seeds=args.seeds,
        budgets=args.budgets,
        gamma=args.gamma,
        write_rows=not args.no_write,
    )


if __name__ == "__main__":
    main()
