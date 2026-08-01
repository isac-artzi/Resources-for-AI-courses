"""
train/train.py — the one command that reproduces every number in the README.

    python -m train.train

It runs the three stages in the order their dependencies demand, and nothing
else lives here: each stage is a module you can run on its own while you are
working on it. The reason this file exists at all is reproducibility. A README
that says "run these four scripts, in this order, with these flags" is a README
whose results nobody will reproduce, including its author six weeks later.

Stages
  1. train/value_iteration.py   plan exactly, log one row per sweep, export
  2. train/monte_carlo.py       learn from samples, log every episode, export
  3. train/compare.py           10 seeds x 6 budgets, RMSE + confidence
                                intervals + the equivalence test

Runtime is about 25 seconds on a laptop and it needs no GPU, which is worth
saying out loud: the expensive part of this course is not the arithmetic, it is
the plumbing. Use --quick while you are iterating on the plumbing.

Nothing here writes to Postgres directly. Every row goes through
`shared/store.py`, so with no credentials configured the whole pipeline still
runs end to end against the in-process fallback and the artifacts still land in
policies/. That is deliberate: a training script that cannot run without a
database is a training script you cannot debug on a train.
"""

from __future__ import annotations

import argparse
import time

from train import compare, monte_carlo, value_iteration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="a tenth of the budget everywhere. Fine for smoke-testing the "
             "pipeline; NOT the configuration whose numbers you report. It "
             "OVERWRITES the committed artifacts and the committed convergence "
             "report, so re-run without it before you commit.",
    )
    parser.add_argument("--seeds", type=int, default=compare.DEFAULT_SEEDS)
    args = parser.parse_args()

    scale = 10 if args.quick else 1
    t0 = time.perf_counter()

    print("=" * 72)
    print("1/3  value iteration — the exact solution")
    print("=" * 72)
    plan, _ = value_iteration.run()

    print()
    print("=" * 72)
    print("2/3  monte carlo — control with exploring starts, then evaluation")
    print("=" * 72)
    monte_carlo.run(
        control_episodes=monte_carlo.DEFAULT_CONTROL_EPISODES // scale,
        eval_episodes=monte_carlo.DEFAULT_EVAL_EPISODES // scale,
        seed=0,
    )

    print()
    print("=" * 72)
    print("3/3  convergence study — how much experience buys how much accuracy")
    print("=" * 72)
    budgets = tuple(max(b // scale, 10) for b in compare.DEFAULT_BUDGETS)
    summary = compare.run_convergence_study(seeds=args.seeds, budgets=budgets)

    print()
    print("=" * 72)
    print(f"done in {time.perf_counter() - t0:.1f}s")
    print(f"  exact V(start)             {plan.V[0]:.6f}")
    print("  artifacts                  policies/value_iteration.npz, "
          "policies/monte_carlo.npz")
    print(f"  convergence report         {compare.REPORT_PATH}")
    n = summary["episodes_to_indistinguishable"]
    print(f"  indistinguishable from     {n} episodes"
          if n is not None else
          "  never became indistinguishable within the budget grid — report that")
    if args.quick:
        print("\n  --quick was used. These numbers are a smoke test, not a result.")


if __name__ == "__main__":
    main()
