"""
train/train.py — the one command that produces the three deployable agents.

TRAINING TIER (imports torch, transitively). Runs on your laptop or in Colab.

    python -m train.train                 # the report budget
    python -m train.train --quick         # the sandbox budget: minutes, not hours
    python -m train.train --only sac      # retrain one agent without touching the others

What it does, for each of A2C, PPO and SAC in turn:

    1. Trains one run, writing `experiments`, `episodes` and `policy_updates`
       rows as it goes.
    2. Evaluates the trained policy deterministically and writes an
       `evaluations` row.
    3. Exports the actor's weights to `policies/<name>.npz` and registers the
       artifact in `policies` with its size and checksum.

Point 3 is where the two tiers meet, and the ordering matters: the artifact is
written only after the run it came from exists in the database, so
`policies.experiment_id` points at a real row and "which run produced the thing
we deployed" stays answerable.

This script trains ONE seed per algorithm, because its job is to produce the
three artifacts the service deploys. It is NOT the evidence. The comparisons the
product brief asks for need at least three seeds each and live in
`train/compare.py` (A2C vs PPO at matched step budgets) and
`train/entropy_sweep.py` (SAC under three temperature regimes). A single seed is
a demo; three seeds is a measurement.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np

from envs import ENV_SPECS
from train.export import register
from train.nets import configure_torch_threads

# The two budgets, side by side, so the difference is visible rather than buried
# in a flag. `--quick` exists so that a fresh fork finishes in a couple of
# minutes and you find your bugs before you spend an hour on them; every number
# it produces is a smoke test and not a result. Say in your Quantitative
# Analysis which budget produced the numbers you are quoting.
BUDGETS = {
    "report": {"a2c_steps": 60_000, "ppo_steps": 60_000, "sac_steps": 15_000},
    "quick": {"a2c_steps": 20_000, "ppo_steps": 20_000, "sac_steps": 6_000},
}

ARTIFACTS = {
    "a2c": "policies/a2c_cartpole.npz",
    "ppo": "policies/ppo_acrobot.npz",
    "sac": "policies/sac_pendulum.npz",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train the three Control Bench agents and export them for serving."
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="sandbox budget: short runs that finish in minutes. Not a result.",
    )
    ap.add_argument("--seed", type=int, default=0, help="the seed for all three runs")
    ap.add_argument(
        "--only",
        choices=sorted(ARTIFACTS),
        action="append",
        help="train only these agents (repeatable). Default: all three.",
    )
    ap.add_argument("--a2c-steps", type=int, default=None)
    ap.add_argument("--ppo-steps", type=int, default=None)
    ap.add_argument("--sac-steps", type=int, default=None)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    configure_torch_threads()

    from shared.store import get_store

    budget = BUDGETS["quick" if args.quick else "report"]
    steps = {
        "a2c": args.a2c_steps or budget["a2c_steps"],
        "ppo": args.ppo_steps or budget["ppo_steps"],
        "sac": args.sac_steps or budget["sac_steps"],
    }
    wanted = sorted(set(args.only or list(ARTIFACTS)))
    store = get_store()
    log_every = 0 if args.quiet else 50
    summary: dict[str, Any] = {"budget": "quick" if args.quick else "report", "agents": {}}

    for name in wanted:
        out = pathlib.Path(ARTIFACTS[name])
        if name == "a2c":
            from train.a2c import A2CConfig, evaluate, train_a2c

            cfg = A2CConfig(total_steps=steps["a2c"], seed=args.seed)
            run = train_a2c(cfg, store=store, log_every=log_every)
            ev = evaluate(run.actor, cfg.env_id, args.eval_episodes)
            row = run.actor.export(out, env_id=cfg.env_id)
        elif name == "ppo":
            from train.ppo import PPOConfig, evaluate, train_ppo

            cfg = PPOConfig(total_steps=steps["ppo"], seed=args.seed)
            run = train_ppo(cfg, store=store, log_every=log_every)
            ev = evaluate(run.actor, cfg.env_id, args.eval_episodes)
            row = run.actor.export(out, env_id=cfg.env_id)
        else:
            from train.sac import SACConfig, evaluate, train_sac

            cfg = SACConfig(total_steps=steps["sac"], seed=args.seed)
            run = train_sac(cfg, store=store, log_every=log_every)
            ev = evaluate(run.actor, cfg.env_id, min(args.eval_episodes, 10))
            row = run.actor.export(out, env_id=cfg.env_id)

        store.insert_evaluation(
            {
                "experiment_id": run.experiment_id,
                "at_training_episode": len(run.episode_returns),
                **ev,
            }
        )
        register(row, experiment_id=run.experiment_id)

        spec = ENV_SPECS[cfg.env_id]
        summary["agents"][name] = {
            "env_id": cfg.env_id,
            "experiment_id": run.experiment_id,
            "env_steps": steps[name],
            "episodes": len(run.episode_returns),
            "train_mean_return_last_100": run.mean_return_last(100),
            # Both windows, because on a short run they answer different
            # questions. SAC's first 1,000 steps are UNIFORMLY RANDOM actions by
            # design, so on a 6,000-step run a third of the episodes are not the
            # agent at all and the 100-episode mean is dragged towards the random
            # baseline. The 10-episode tail is what the policy is doing now.
            # Quote whichever you like — and quote the window with it.
            "train_mean_return_last_10": run.mean_return_last(10),
            # Quoted WITH the random baseline, always. "−92" means nothing until
            # the reader knows the floor is −500, and this is the one place the
            # two numbers are guaranteed to appear together.
            "random_baseline": spec.random_return,
            "eval_mean_return": ev["mean_return"],
            "eval_stderr": ev["stderr_return"],
            "artifact": str(out),
            "artifact_bytes": row["bytes"],
            "artifact_sha256": row["sha256"],
            "obs_dim": row["obs_dim"],
            "n_actions": row["n_actions"],
            "action_space": row["action_space"],
        }
        if not args.quiet:
            print(
                f"\n[{name}] {cfg.env_id}: "
                f"train mean100={run.mean_return_last(100):.1f} "
                f"(random ≈ {spec.random_return:.0f})  "
                f"eval={ev['mean_return']:.1f} ± {ev['stderr_return']:.1f}  "
                f"→ {out} ({row['bytes']} bytes)\n",
                flush=True,
            )

    print(json.dumps(summary, indent=2, default=_jsonable))

    if not _data_tier_configured():
        print(
            "\nNOTE: SUPABASE_URL is unset, so every row this run wrote went to the "
            "in-process fallback store and vanished when the run ended. The .npz files "
            "on disk are real; the telemetry is not. Fill in .env before the run you "
            "intend to report."
        )
    return summary


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return str(v)


def _data_tier_configured() -> bool:
    from shared.config import get_settings

    return get_settings().data_tier_configured


if __name__ == "__main__":  # pragma: no cover - a CLI entry point
    main()
