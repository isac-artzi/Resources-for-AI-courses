"""
train/compare.py — A2C versus PPO at MATCHED environment-step budgets.

TRAINING TIER (imports torch, transitively).

    python -m train.compare --quick                 # sandbox: a few minutes
    python -m train.compare --steps 60000 --seeds 3 # the budget you report

What "matched budget" means, and why it is the only fair axis
--------------------------------------------------------------
Both algorithms are run on BOTH discrete environments for exactly the same
number of environment steps. Not the same number of episodes, and not the same
wall clock:

  * **Episodes are not comparable.** An untrained CartPole agent's episodes last
    20 steps and a trained one's last 500. Two runs at "episode 200" have
    consumed wildly different amounts of experience, and an agent that learns
    faster is penalised on that axis for having longer episodes. On Acrobot the
    bias runs the other way. Comparing at matched episodes is how a
    sample-efficiency claim gets accidentally inverted.
  * **Wall clock is a different question.** PPO takes ~40 gradient steps per
    1,024 environment steps; A2C takes 8. PPO is therefore slower per
    environment step and — this is the finding — usually further ahead at any
    given step count, because it extracts more from each transition. If your
    constraint is compute, measure compute and say so; if your constraint is
    interaction with the world (a robot, a metered simulator, a human), steps is
    the axis that bills you.

Both algorithms run on both environments, rather than each on "its own" task,
because a comparison in which each method gets a different problem is not a
comparison. CartPole is where A2C is competitive; Acrobot is where the
difference shows. Report both, including the one that does not flatter your
preferred method.

The deliverable is a 150–250 word summary of what the comparison shows AND WHAT
IT DOES NOT. Two things it does not show, for free: that PPO is better in
general (two classic-control tasks at one budget is not "in general"), and that
the gap would survive hyperparameter tuning of the losing arm (nobody tuned A2C
here; both use this repository's defaults, which is a deliberate control and
also a limitation).
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np

from envs import spec_for
from train.a2c import A2CConfig, train_a2c
from train.nets import configure_torch_threads
from train.onpolicy import evaluate_discrete
from train.ppo import PPOConfig, train_ppo

DISCRETE_ENVS = ["CartPole-v1", "Acrobot-v1"]


def run_comparison(
    envs: list[str] | None = None,
    steps: int = 60_000,
    seeds: int = 3,
    seed_start: int = 0,
    eval_episodes: int = 20,
    quiet: bool = False,
) -> list[dict[str, Any]]:
    """Train every (algorithm × environment × seed) cell at the same step budget."""
    from shared.store import get_store

    store = get_store()
    envs = envs or DISCRETE_ENVS
    results: list[dict[str, Any]] = []

    for env_id in envs:
        spec = spec_for(env_id)
        for algo in ("a2c", "ppo"):
            for k in range(seeds):
                seed = seed_start + k
                if algo == "a2c":
                    cfg = A2CConfig(env_id=env_id, total_steps=steps, seed=seed)
                    run = train_a2c(cfg, store=store, log_every=0 if quiet else 100)
                else:
                    cfg = PPOConfig(env_id=env_id, total_steps=steps, seed=seed)
                    run = train_ppo(cfg, store=store, log_every=0 if quiet else 100)

                ev = evaluate_discrete(run.actor, env_id, eval_episodes)
                store.insert_evaluation(
                    {
                        "experiment_id": run.experiment_id,
                        "at_training_episode": len(run.episode_returns),
                        **ev,
                    }
                )
                results.append(
                    {
                        "algorithm": algo,
                        "env_id": env_id,
                        "seed": seed,
                        "experiment_id": run.experiment_id,
                        "env_steps": steps,
                        "episodes": len(run.episode_returns),
                        "mean_return_last_100": run.mean_return_last(100),
                        "eval_mean_return": ev["mean_return"],
                        "eval_stderr": ev["stderr_return"],
                        # The sample-efficiency headline: how much experience did
                        # this arm need to clear the bar, measured in STEPS.
                        "steps_to_threshold": _steps_to_threshold(run, spec.threshold),
                        "threshold": spec.threshold,
                        "random_baseline": spec.random_return,
                        # PPO only. Carried into the comparison table so the
                        # trust-region column sits next to the performance it
                        # was supposed to protect.
                        "mean_kl": _mean_of(run.updates, "kl_divergence"),
                        "mean_clip_fraction": _mean_of(run.updates, "clip_fraction"),
                        "gradient_steps": _gradient_steps(algo, cfg, steps),
                    }
                )
                if not quiet:
                    print(
                        f"  [{algo} {env_id} seed={seed}] "
                        f"last100={results[-1]['mean_return_last_100']:8.1f}  "
                        f"eval={ev['mean_return']:8.1f}  "
                        f"steps_to_{spec.threshold:g}={results[-1]['steps_to_threshold']}",
                        flush=True,
                    )
    return results


def _steps_to_threshold(run, threshold: float, window: int = 10) -> int | None:
    """Environment steps consumed before the trailing-window mean reached `threshold`.

    Steps rather than episodes, for the reason in the module docstring. None
    when the bar was never cleared — a result, not a missing value, and not a
    sentinel to be averaged.
    """
    ep = run.episodes_to_threshold(threshold, window)
    if ep is None:
        return None
    return int(run.episode_steps[ep]) if ep < len(run.episode_steps) else None


def _mean_of(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def _gradient_steps(algo: str, cfg, steps: int) -> int:
    """How many optimiser steps each arm took for the same environment budget.

    Reported because it is the other half of the trade. PPO is ahead at matched
    STEPS partly because it takes roughly five times as many gradient steps on
    the same data; a reader who only sees the sample-efficiency chart will
    conclude PPO is better full stop, and this column is what stops that.
    """
    iterations = steps // cfg.n_steps
    if algo == "a2c":
        return iterations
    minibatches = max(1, cfg.n_steps // cfg.minibatch_size)
    return iterations * cfg.epochs * minibatches


def summarise(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate across seeds. One row per (algorithm, environment)."""
    out: list[dict[str, Any]] = []
    keys = sorted({(r["env_id"], r["algorithm"]) for r in results})
    for env_id, algo in keys:
        cell = [r for r in results if r["env_id"] == env_id and r["algorithm"] == algo]
        finals = np.asarray([r["mean_return_last_100"] for r in cell], dtype=np.float64)
        evals = np.asarray([r["eval_mean_return"] for r in cell], dtype=np.float64)
        reached = [r["steps_to_threshold"] for r in cell if r["steps_to_threshold"] is not None]
        out.append(
            {
                "env_id": env_id,
                "algorithm": algo,
                "seeds": len(cell),
                "env_steps": cell[0]["env_steps"],
                "gradient_steps": cell[0]["gradient_steps"],
                "mean_return_last_100": float(finals.mean()),
                # Standard error ACROSS SEEDS. The only error bar that supports
                # a claim about a configuration; the spread within one run says
                # nothing about whether the arm would work again.
                "stderr_across_seeds": float(
                    finals.std(ddof=1) / np.sqrt(len(finals)) if len(finals) > 1 else 0.0
                ),
                "eval_mean_return": float(evals.mean()),
                "reached_threshold": len(reached),
                "mean_steps_to_threshold": float(np.mean(reached)) if reached else None,
                "threshold": cell[0]["threshold"],
                "random_baseline": cell[0]["random_baseline"],
                "mean_kl": _mean_of(cell, "mean_kl"),
            }
        )
    return out


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description="A2C vs PPO at matched environment-step budgets")
    ap.add_argument("--steps", type=int, default=60_000, help="the MATCHED budget, per run")
    ap.add_argument("--quick", action="store_true", help="20,000 steps — a smoke test, not a result")
    ap.add_argument("--seeds", type=int, default=3, help="fewer than 3 is not evidence")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--envs", nargs="*", default=None, choices=DISCRETE_ENVS)
    ap.add_argument("--out", default=None, help="also write the results to this JSON file")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    configure_torch_threads()

    steps = 20_000 if args.quick else args.steps
    if args.seeds < 3:
        print(
            f"WARNING: {args.seeds} seed(s). A policy gradient on CartPole can reach 500 "
            "or plateau at 40 with nothing changed but the initialisation, so below "
            "three seeds you are measuring the seed rather than the algorithm.",
            flush=True,
        )

    results = run_comparison(
        envs=args.envs,
        steps=steps,
        seeds=args.seeds,
        seed_start=args.seed_start,
        quiet=args.quiet,
    )
    summary = {"matched_env_steps": steps, "seeds": args.seeds, "cells": summarise(results)}
    print(json.dumps(summary, indent=2))
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps({"runs": results, **summary}, indent=2))
    return summary


if __name__ == "__main__":  # pragma: no cover - a CLI entry point
    main()
