"""
train/train.py — the one command that produces a deployable agent.

TRAINING TIER (imports torch, transitively). Runs on your laptop or in Colab.

    python -m train.train                       # the defaults below
    python -m train.train --episodes 1000 --seeds 3
    python -m train.train --no-baseline         # the ablation's other arm, alone

What it does, in order:

    1. Trains `--seeds` independent runs of ONE configuration, writing
       `experiments`, `episodes` and `gradient_stats` rows as it goes.
    2. Evaluates each trained policy greedily and writes an `evaluations` row.
    3. Exports the BEST seed's weights to `policies/<name>.npz` and registers
       the artifact in `policies` with its size and checksum.

Point 3 is where the two tiers meet, and the ordering matters: the artifact is
written only after the run it came from exists in the database, so the
`policies.experiment_id` foreign key points at a real row and "which run
produced the thing we deployed" stays answerable.

For the 2x2 comparison the product brief actually asks for, use
`python -m train.ablation` — this script trains one cell.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from train.export import register
from train.vpg import RunResult, VPGConfig, evaluate, train_vpg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train a VPG agent and export it for serving.")
    ap.add_argument("--episodes", type=int, default=400,
                    help="training episodes per seed (the report budget is 1000)")
    ap.add_argument("--seeds", type=int, default=3,
                    help="how many independent seeds; fewer than 3 is not evidence")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--policy-lr", type=float, default=VPGConfig.policy_lr)
    ap.add_argument("--batch-episodes", type=int, default=VPGConfig.batch_episodes)
    ap.add_argument("--entropy-coef", type=float, default=VPGConfig.entropy_coef)
    # `--no-baseline` / `--importance-sampling` rather than `--baseline=false`.
    # A flag that must be parsed as a boolean string is a flag that will one day
    # be passed "False" and read as True.
    ap.add_argument("--no-baseline", action="store_true", help="ablate the value baseline")
    ap.add_argument("--importance-sampling", action="store_true",
                    help="reuse the previous batch off-policy")
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--out", default="policies/vpg_cartpole.npz",
                    help="artifact path; the stem is the name /act will accept")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    from shared.store import get_store

    store = get_store()

    results: list[tuple[RunResult, dict]] = []
    for k in range(args.seeds):
        cfg = VPGConfig(
            episodes=args.episodes,
            seed=args.seed_start + k,
            policy_lr=args.policy_lr,
            batch_episodes=args.batch_episodes,
            entropy_coef=args.entropy_coef,
            use_baseline=not args.no_baseline,
            use_importance_sampling=args.importance_sampling,
        )
        run = train_vpg(cfg, store=store, log_every=0 if args.quiet else 100)

        # Greedy evaluation, in its own table. The training curve is what the
        # exploring policy scored; this is what the deployed policy scores.
        ev = evaluate(run.policy, episodes=args.eval_episodes)
        store.insert_evaluation(
            {"experiment_id": run.experiment_id, "at_training_episode": args.episodes, **ev}
        )
        results.append((run, ev))
        if not args.quiet:
            print(
                f"[{cfg.arm} seed={cfg.seed}] done — "
                f"train mean50={run.mean_return_last(50):.1f}  "
                f"greedy eval={ev['mean_return']:.1f} +/- {ev['stderr_return']:.1f}",
                flush=True,
            )

    # Selected on GREEDY EVALUATION return, not on the training curve. Picking
    # on the training curve rewards a seed that got lucky in its last few noisy
    # episodes, and the artifact you deploy should be the one that is best at
    # the thing you will deploy it to do.
    #
    # Be honest about this in your report: choosing the best of N seeds is a
    # selection effect, and the greedy return of the SELECTED seed is an
    # optimistic estimate of what a fresh seed would give you. Report the
    # across-seed mean too — it is in `evaluations` for exactly this reason.
    best_run, best_eval = max(results, key=lambda pair: pair[1]["mean_return"])

    out = pathlib.Path(args.out)
    row = best_run.policy.export(out)
    register(row, experiment_id=best_run.experiment_id)

    across = np.asarray([e["mean_return"] for _, e in results], dtype=np.float64)
    summary = {
        "artifact": str(out),
        "artifact_bytes": row["bytes"],
        "artifact_sha256": row["sha256"],
        "arm": best_run.config.arm,
        "episodes_per_seed": args.episodes,
        "seeds": args.seeds,
        "selected_seed": best_run.config.seed,
        "selected_experiment_id": best_run.experiment_id,
        "selected_greedy_return": best_eval["mean_return"],
        "across_seed_greedy_mean": float(across.mean()),
        "across_seed_greedy_stderr": float(
            across.std(ddof=1) / np.sqrt(len(across)) if len(across) > 1 else 0.0
        ),
        "train_mean_return_last_50": best_run.mean_return_last(50),
        "explained_variance": best_run.explained_variance,
    }
    print(json.dumps(summary, indent=2))

    if not get_store_configured():
        print(
            "\nNOTE: SUPABASE_URL is unset, so every row this run wrote went to the "
            "in-process fallback store and vanished when the run ended. The .npz on "
            "disk is real; the telemetry is not. Fill in .env before the run you "
            "intend to report."
        )
    return summary


def get_store_configured() -> bool:
    from shared.config import get_settings

    return get_settings().data_tier_configured


if __name__ == "__main__":  # pragma: no cover - a CLI entry point
    main()
