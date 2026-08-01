"""
train/entropy_sweep.py — SAC under three temperature regimes, ≥ 3 seeds each.

TRAINING TIER (imports torch, transitively).

    python -m train.entropy_sweep --quick          # sandbox: ~10 minutes
    python -m train.entropy_sweep --seeds 3        # the budget you report (20,000 steps)

The three arms, and what each one is testing
--------------------------------------------
    α = 0.5    high temperature. Entropy is worth a lot relative to reward, so
               the optimal policy of the maximum-entropy objective is genuinely
               spread out. On Pendulum, whose per-step reward reaches −16, an α
               of 0.5 is large: the agent is being paid to be uncertain at a
               price comparable to the cost of being wrong.
    α = 0.01   low temperature. Nearly the un-regularised objective. Converges
               to something deterministic quickly and can converge to something
               deterministic and WRONG, because it stops exploring before it has
               seen the swing-up.
    auto       α is a Lagrange multiplier solved for a target entropy of
               −dim(A) = −1. It starts at 0.2 and moves.

What is reported, and why each column exists
--------------------------------------------
    convergence speed   `episodes_to_threshold` — the first episode whose
                        trailing 10-episode mean reached the stated threshold.
                        NULL when the run never got there, which is a result and
                        must not be encoded as a large number.
    final performance   `mean_return_last_100` — mean training return over the
                        last 100 episodes, or over all of them if the run was
                        shorter, in which case `episodes` says so.

                        MIND THE BUDGET HERE. A Pendulum episode is exactly 200
                        steps, so a 15,000-step run has 75 episodes and the
                        "last 100" window is the WHOLE RUN — including the first
                        1,000 steps of uniformly random actions. The number is
                        then a mix of the agent and its warm-up, and it drags
                        every arm towards the random baseline equally, which
                        makes the arms look closer together than they are. The
                        default below is 20,000 steps for exactly that reason:
                        100 episodes is the smallest budget at which the column
                        means what its name says. `run_sweep` warns if a run
                        finishes with fewer.
    stability           the VARIANCE ACROSS SEEDS of that final performance,
                        computed by `summarise()` below. Note that this is a
                        different quantity from `return_std_last_100`, which is
                        the spread WITHIN one run. A configuration can be steady
                        within a seed and wildly seed-dependent, and only the
                        first of those two numbers would notice.
    exploration         `mean_policy_entropy` — the mean differential entropy of
                        the actor over the run's updates. This is the column
                        that shows α = 0.5 exploring more than α = 0.01, and it
                        is NEGATIVE for all of them, which is normal: a
                        continuous density can exceed 1, so differential entropy
                        is not bounded below by zero.

A warning about the honest reading. Three seeds is enough to notice a large
effect and not enough to measure a small one. If two arms' means differ by less
than the spread across their seeds, the correct sentence is "this study did not
separate them", not "α = 0.5 was slightly better". Say which one your numbers
support.
"""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING, Any

import numpy as np

from envs import spec_for

if TYPE_CHECKING:  # pragma: no cover - annotations only; never imported at runtime
    from train.sac import SACRunResult

# NOTE THE MISSING IMPORT. `train.sac` and `train.nets` are imported INSIDE the
# functions that need them, not at module scope, and that is deliberate:
# `summarise()` below is pure arithmetic over dictionaries, and
# tests/test_topic4_tables.py checks it in the ordinary test process — the one
# that `tests/test_no_torch.py` requires to stay free of PyTorch. A module-scope
# `from train.sac import ...` would drag torch into `sys.modules` the moment that
# test module was collected, and the no-torch guard would start failing
# depending on pytest's collection order. Same reasoning as `train/export.py`.

# The three arms of the sweep. (label, auto?, alpha) — the label is what lands
# in `entropy_sweep.alpha_setting` and what the Streamlit table groups on, so it
# is written once here rather than reconstructed from the numbers downstream.
ARMS: list[tuple[str, bool, float]] = [
    ("alpha=0.5", False, 0.5),
    ("alpha=0.01", False, 0.01),
    ("auto", True, 0.2),          # 0.2 is the INITIAL value; the tuner moves it
]


def run_sweep(
    steps: int = 20_000,
    seeds: int = 3,
    seed_start: int = 0,
    threshold: float | None = None,
    eval_episodes: int = 10,
    quiet: bool = False,
) -> list[dict[str, Any]]:
    """Train every (arm × seed) cell and write one `entropy_sweep` row each."""
    from shared.store import get_store
    from train.sac import SACConfig, evaluate, train_sac

    store = get_store()
    spec = spec_for("Pendulum-v1")
    threshold = spec.threshold if threshold is None else threshold
    rows: list[dict[str, Any]] = []

    for label, auto, alpha in ARMS:
        for k in range(seeds):
            cfg = SACConfig(
                total_steps=steps,
                seed=seed_start + k,
                alpha=alpha,
                auto_alpha=auto,
            )
            run = train_sac(cfg, store=store, log_every=0 if quiet else 25)
            ev = evaluate(run.actor, cfg.env_id, eval_episodes)
            store.insert_evaluation(
                {
                    "experiment_id": run.experiment_id,
                    "at_training_episode": len(run.episode_returns),
                    **ev,
                }
            )
            row = _sweep_row(run, label, auto, threshold, ev)
            store.insert_entropy_sweep(row)
            rows.append(row)
            if row["episodes"] < 100:
                # Printed once per run rather than once per sweep, because the
                # number that is wrong is this run's. Silence here would let a
                # short budget produce a column called `mean_return_last_100`
                # that is really "mean over the whole run, warm-up included" —
                # and nothing downstream could tell the difference.
                print(
                    f"  NOTE: this run produced {row['episodes']} episodes, so "
                    "`mean_return_last_100` is the mean over the WHOLE run, including "
                    "the uniformly random start_steps. Use >= 20,000 steps for the "
                    "sweep you report.",
                    flush=True,
                )
            if not quiet:
                print(
                    f"  [{label} seed={cfg.seed}] "
                    f"final={row['mean_return_last_100']:8.1f}  "
                    f"H={row['mean_policy_entropy']:+.3f}  "
                    f"alpha_end={row['alpha_value']:.4f}  "
                    f"to_threshold={row['episodes_to_threshold']}",
                    flush=True,
                )
    return rows


def _sweep_row(
    run: "SACRunResult", label: str, auto: bool, threshold: float, ev: dict[str, Any]
) -> dict[str, Any]:
    return {
        "experiment_id": run.experiment_id,
        "mode": "auto" if auto else "fixed",
        "alpha_setting": label,
        # The FINAL α under automatic tuning, the fixed one otherwise. Where the
        # tuner ended up relative to the two hand-chosen values is the most
        # interesting single number in the sweep, and storing only the setting
        # would lose it.
        "alpha_value": run.final_alpha,
        "seed": run.config.seed,
        "episodes": len(run.episode_returns),
        "env_steps": run.config.total_steps,
        "episodes_to_threshold": run.episodes_to_threshold(threshold),
        "threshold": threshold,
        "mean_return_last_100": run.mean_return_last(100),
        "return_std_last_100": run.std_return_last(100),
        "mean_policy_entropy": run.mean_policy_entropy,
        "eval_mean_return": ev["mean_return"],
    }


def summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate the per-seed rows into the comparison table the brief asks for.

    Computed here as well as in SQL (`entropy_sweep_summary` in
    002_topic4.sql) so that the number in your README and the number a reader
    gets from your database come from the same definition. If they ever
    disagree, one of the two has a bug and you want to find out from a test
    rather than from a grader.

    `episodes_to_threshold` is averaged over the runs that REACHED it, and
    `reached` reports how many did. Averaging over all runs would require a
    value for the ones that never got there, and every choice of value is a lie:
    a large sentinel invents a slow convergence, and dropping the arm entirely
    hides that it failed. Two columns, and the reader can see both.
    """
    out: list[dict[str, Any]] = []
    for label in [a[0] for a in ARMS]:
        arm = [r for r in rows if r["alpha_setting"] == label]
        if not arm:
            continue
        finals = np.asarray([r["mean_return_last_100"] for r in arm], dtype=np.float64)
        reached = [r["episodes_to_threshold"] for r in arm if r["episodes_to_threshold"] is not None]
        out.append(
            {
                "alpha_setting": label,
                "mode": arm[0]["mode"],
                "seeds": len(arm),
                "mean_final_return": float(finals.mean()),
                # ddof=1: these are n samples used to estimate a population
                # variance, and with n = 3 the difference between dividing by 3
                # and by 2 is 50%. Two arms compared with different ddof is a
                # fake finding.
                "across_seed_std": float(finals.std(ddof=1)) if len(finals) > 1 else 0.0,
                "across_seed_variance": float(finals.var(ddof=1)) if len(finals) > 1 else 0.0,
                "mean_policy_entropy": float(
                    np.mean([r["mean_policy_entropy"] for r in arm])
                ),
                "mean_final_alpha": float(np.mean([r["alpha_value"] for r in arm])),
                "reached_threshold": len(reached),
                "mean_episodes_to_threshold": float(np.mean(reached)) if reached else None,
                "threshold": arm[0]["threshold"],
                "mean_eval_return": float(np.mean([r["eval_mean_return"] for r in arm])),
            }
        )
    return out


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description="SAC entropy sweep: 3 temperature regimes x N seeds")
    ap.add_argument("--steps", type=int, default=20_000, help="environment steps per run")
    ap.add_argument(
        "--quick",
        action="store_true",
        help="6,000 steps per run — nine runs in roughly ten minutes. A smoke test.",
    )
    ap.add_argument("--seeds", type=int, default=3, help="fewer than 3 is not evidence")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="convergence bar; defaults to the env spec's stated threshold (-300)",
    )
    ap.add_argument("--out", default=None, help="also write the summary to this JSON file")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    from train.nets import configure_torch_threads

    configure_torch_threads()

    steps = 6_000 if args.quick else args.steps
    if args.seeds < 3:
        # A warning rather than an error: you will want one seed while you are
        # debugging. But the brief asks for three, and a two-seed "comparison"
        # cannot distinguish an arm from an initialisation.
        print(
            f"WARNING: {args.seeds} seed(s). Below three you are measuring the seed, "
            "not the temperature. Fine for a smoke test; not a result.",
            flush=True,
        )

    rows = run_sweep(
        steps=steps,
        seeds=args.seeds,
        seed_start=args.seed_start,
        threshold=args.threshold,
        quiet=args.quiet,
    )
    summary = {
        "env_id": "Pendulum-v1",
        "steps_per_run": steps,
        "seeds": args.seeds,
        "random_baseline": spec_for("Pendulum-v1").random_return,
        "arms": summarise(rows),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        import pathlib

        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps({"rows": rows, **summary}, indent=2))
    return summary


if __name__ == "__main__":  # pragma: no cover - a CLI entry point
    main()
