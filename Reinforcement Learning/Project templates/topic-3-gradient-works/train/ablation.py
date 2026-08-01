"""
train/ablation.py — the 2x2 the product brief is built around.

TRAINING TIER (imports torch, transitively).

    baseline OFF / IS OFF     vanilla policy gradient, the control arm
    baseline ON  / IS OFF     variance reduction from a learned V(s)
    baseline OFF / IS ON      off-policy reuse without variance reduction
    baseline ON  / IS ON      both

Four arms times at least three seeds is twelve runs, and all twelve must be
rows in `experiments`. Three seeds is not a ritual: a single seed of a policy
gradient on CartPole can reach 500 or plateau at 40 with nothing changed but
the initialisation, so a one-seed comparison between arms measures the seed.

    python -m train.ablation                       # sandbox budget, ~minutes
    python -m train.ablation --episodes 1000 --seeds 3   # the report budget

Budget
------
The default here is deliberately SMALL so that the command completes inside a
CI job or a lunch break and you find your bugs cheaply. It is not the budget you
report. The graded run is `--episodes 1000 --seeds 3`; at the small default the
arms have barely separated and any conclusion you draw from it is noise wearing
a chart. Put the real budget's numbers in the README and say which command
produced them.

Reading the result honestly
---------------------------
One trap worth naming before you plot anything. Gradient variance in this
environment grows with the return, because CartPole's return IS the episode
length and a longer episode contributes more terms to the sum. So an arm that
learns faster will show HIGHER raw gradient variance later in training — not
because its estimator is worse but because it is standing somewhere else. Two
defensible ways to compare, both of which belong in your report:

  * compare at the same UPDATE INDEX early in training, before the arms have
    separated in return; and
  * compare at matched return, or normalise by the batch's mean return.

The controlled single-batch measurement in
`train.vpg.compare_baseline_variance` — same trajectories, same parameters,
only the advantage changes — has neither problem, and is the number to lead
with when someone asks whether the baseline works.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import time

import numpy as np

from train.vpg import VPGConfig, evaluate, train_vpg

# The 2x2. Written as a product of two boolean axes rather than four hand-listed
# configurations so that adding a third axis later (entropy on/off, say) is a
# one-line change and cannot silently drop a cell.
AXES = {"use_baseline": (False, True), "use_importance_sampling": (False, True)}


def arms() -> list[dict[str, bool]]:
    keys = list(AXES)
    return [dict(zip(keys, combo)) for combo in itertools.product(*AXES.values())]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the 2x2 baseline x importance-sampling ablation.")
    ap.add_argument("--episodes", type=int, default=120,
                    help="episodes per run. SANDBOX DEFAULT — the report budget is 1000.")
    ap.add_argument("--seeds", type=int, default=3,
                    help="seeds per arm; the rubric floor is 3")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--batch-episodes", type=int, default=VPGConfig.batch_episodes)
    ap.add_argument("--policy-lr", type=float, default=VPGConfig.policy_lr)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--out", default="", help="optional path to write the summary JSON")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    from shared.store import get_store

    store = get_store()

    if args.seeds < 3:
        # A warning, not a refusal. You will legitimately want one seed while
        # debugging; you must not legitimately report one.
        print(
            f"WARNING: {args.seeds} seed(s) per arm. The rubric requires at least 3, "
            "and below 3 you cannot separate the arm's effect from the seed's.",
            flush=True,
        )

    rows = []
    started = time.time()
    for arm in arms():
        for k in range(args.seeds):
            cfg = VPGConfig(
                episodes=args.episodes,
                seed=args.seed_start + k,
                batch_episodes=args.batch_episodes,
                policy_lr=args.policy_lr,
                **arm,
            )
            run = train_vpg(cfg, store=store)
            ev = evaluate(run.policy, episodes=args.eval_episodes)
            store.insert_evaluation(
                {"experiment_id": run.experiment_id, "at_training_episode": args.episodes, **ev}
            )

            on_policy = [s for s in run.gradient_stats if not s["off_policy"]]
            rows.append(
                {
                    "arm": cfg.arm,
                    "use_baseline": cfg.use_baseline,
                    "use_importance_sampling": cfg.use_importance_sampling,
                    "seed": cfg.seed,
                    "experiment_id": run.experiment_id,
                    "train_mean_return_last_50": run.mean_return_last(50),
                    "greedy_mean_return": ev["mean_return"],
                    "greedy_stderr": ev["stderr_return"],
                    # The MEDIAN, not the mean. The distribution of gradient
                    # variance across updates is heavy-tailed — one update
                    # during which the policy happened to sit near a decision
                    # boundary can be two orders of magnitude above the rest —
                    # and a mean over that is a report about one update.
                    "median_gradient_variance": float(
                        np.median([s["gradient_variance"] for s in on_policy])
                    ),
                    "final_policy_entropy": float(on_policy[-1]["policy_entropy"]),
                    "explained_variance": run.explained_variance,
                    "updates": len(run.gradient_stats),
                }
            )
            if not args.quiet:
                r = rows[-1]
                print(
                    f"{r['arm']:>20s} seed={r['seed']}  "
                    f"train50={r['train_mean_return_last_50']:6.1f}  "
                    f"greedy={r['greedy_mean_return']:6.1f}  "
                    f"med-gvar={r['median_gradient_variance']:.4g}  "
                    f"H={r['final_policy_entropy']:.3f}",
                    flush=True,
                )

    summary = {
        "episodes_per_run": args.episodes,
        "seeds_per_arm": args.seeds,
        "runs": rows,
        "by_arm": _aggregate(rows),
        "wall_clock_seconds": round(time.time() - started, 1),
        "note": (
            "Every row above is also a row in `experiments`; this JSON is a "
            "convenience, not the record. Regenerate the comparison table in your "
            "README with a GROUP BY against the database, so a reader can check it."
        ),
    }
    print(json.dumps(summary["by_arm"], indent=2))
    if args.out:
        # `runs/` is gitignored, deliberately: this JSON is a convenience for the
        # next twenty minutes, and the record of the experiment is the database.
        # mkdir here so `make report-budget` does not lose two hours of training
        # to a missing directory on its final line.
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
    return summary


def _aggregate(rows: list[dict]) -> list[dict]:
    """Group by arm, with a standard error. Never a mean without one."""
    out = []
    for arm in sorted({r["arm"] for r in rows}):
        mine = [r for r in rows if r["arm"] == arm]
        g = np.asarray([r["greedy_mean_return"] for r in mine], dtype=np.float64)
        out.append(
            {
                "arm": arm,
                "seeds": len(mine),
                "greedy_mean": float(g.mean()),
                # Across SEEDS, which is the only error bar that supports a
                # claim about the configuration. The per-seed stderr in
                # `evaluations` is across evaluation episodes and answers a
                # different question — quoting it here would understate the
                # uncertainty by roughly the amount that makes a null result
                # look significant.
                "greedy_stderr_across_seeds": float(
                    g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0.0
                ),
                "median_gradient_variance": float(
                    np.median([r["median_gradient_variance"] for r in mine])
                ),
                "mean_final_entropy": float(np.mean([r["final_policy_entropy"] for r in mine])),
            }
        )
    return out


if __name__ == "__main__":  # pragma: no cover - a CLI entry point
    main()
