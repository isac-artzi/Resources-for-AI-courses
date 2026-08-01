"""
train/train.py — the one command the README quickstart runs.

    python -m train.train                          # the product configuration
    python -m train.train --episodes 4000 --seeds 2 --random-episodes 200   # smoke test

It does four things, in this order, because each depends on the last:

  1. runs the random baseline on every seed          -> `experiments`, `episodes`
  2. runs Q-learning on the same seeds               -> `experiments`, `episodes`,
                                                        `evaluations`
  3. exports the BEST seed's Q-table to policies/    -> `train.export.export_qtable`
  4. registers the artifact with its size and SHA    -> `policies`

Three points worth understanding before you change it.

**Both agents, same seeds, same tables.** The baseline is not a formality. It is
the denominator, and running it under the same seeds and writing it to the same
schema is what makes "the trained agent beats random" a `GROUP BY` rather than
an assertion.

**"Best seed" is a defensible export policy and a dishonest reporting policy.**
Shipping the best of three artifacts is correct — you deploy one policy and you
should deploy the good one. Quoting that seed's score as *the* result is not:
the maximum of three noisy numbers is biased upwards. Export the best, report
the mean across seeds with its spread, and say which you did. This script
prints both so you have no excuse.

**An untrained artifact is exported too.** The "Watch" tab needs something to
contrast the trained agent against, and the honest way to serve "random"
through a typed policy contract is a real registered artifact whose Q-table is
all zeros: with equal values, /rollout's sampling path draws uniformly. The
alternative — a special case in the service tier that means "ignore the
artifact and act randomly" — is untestable and would be the only code path in
the product that is not a policy.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from envs import ENV_ID, MAP_NAME, action_space_size, observation_space_size
from train.export import export_qtable, register
from train.qlearning import evaluate_greedy, q_learning, schedule_preview
from train.random_agent import run_random_agent
from train.telemetry import mean_and_stderr, warn_if_data_tier_is_local

UNTRAINED_NAME = "untrained_policy"
TRAINED_NAME = "q_table"


def export_untrained_baseline(policy_dir: pathlib.Path, map_name: str = MAP_NAME) -> dict:
    """Register an all-zero Q-table so the UI can serve an untrained agent.

    Zeros, not random values. A table of random values is a *fixed arbitrary*
    policy — it walks into the same wall every episode, which looks broken
    rather than untrained. Zeros are genuinely indifferent: every action has the
    same value, so the softmax over a row is uniform and the sampled rollout is
    a real random policy. Under the greedy path the same table always returns
    action 0, which is itself a useful thing for a student to observe and
    explain.
    """
    Q = np.zeros((observation_space_size(map_name), action_space_size()), dtype=np.float32)
    row = export_qtable(Q, policy_dir / f"{UNTRAINED_NAME}.npz")
    try:
        register(row, experiment_id=None)
    except Exception as exc:  # noqa: BLE001 — a registry failure must not lose the artifact
        print(f"[train] artifact written but not registered: {exc}")
    return row


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(
        description="Train the Lake Pilot agents end to end and export an artifact."
    )
    p.add_argument("--episodes", type=int, default=20_000,
                   help="Q-learning episodes per seed (syllabus floor: 20,000)")
    p.add_argument("--seeds", type=int, default=3, help="independent seeds (syllabus floor: 3)")
    p.add_argument("--random-episodes", type=int, default=1_000,
                   help="baseline episodes per seed (syllabus asks for 1,000)")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps-schedule", default="linear:1.0:0.05:0.6")
    p.add_argument("--q-init", type=float, default=0.0)
    p.add_argument("--eval-every", type=int, default=2_000)
    p.add_argument("--eval-episodes", type=int, default=200,
                   help="episodes per greedy evaluation; more episodes, tighter standard error")
    p.add_argument("--map", dest="map_name", default=MAP_NAME)
    p.add_argument("--no-slip", action="store_true")
    p.add_argument("--policy-dir", default="policies")
    p.add_argument("--no-log", action="store_true")
    args = p.parse_args(argv)

    slippery = not args.no_slip
    log = not args.no_log
    if log:
        warn_if_data_tier_is_local()

    policy_dir = pathlib.Path(args.policy_dir)
    started = time.time()

    print("=" * 78)
    print(f"Lake Pilot — {ENV_ID} {args.map_name} "
          f"{'slippery' if slippery else 'deterministic'}")
    print(f"{args.seeds} seeds · {args.random_episodes:,} random episodes · "
          f"{args.episodes:,} Q-learning episodes each")
    print("epsilon schedule:", args.eps_schedule,
          "->", schedule_preview(args.eps_schedule, args.episodes))
    print("=" * 78)

    # -- 1. baseline --------------------------------------------------------
    print("\n[1/4] random baseline")
    random_returns: list[float] = []
    for seed in range(args.seeds):
        r = run_random_agent(
            episodes=args.random_episodes, seed=seed,
            map_name=args.map_name, is_slippery=slippery, log=log,
        )
        random_returns.extend(r["returns"])
        print(f"  seed {seed}: mean {r['mean_return']:.4f} ± {r['stderr_return']:.4f}")
    rnd_mean, _, rnd_stderr = mean_and_stderr(random_returns)

    # -- 2. the learner -----------------------------------------------------
    print("\n[2/4] tabular Q-learning")
    runs = []
    for seed in range(args.seeds):
        print(f"  seed {seed}")
        result = q_learning(
            episodes=args.episodes, seed=seed, alpha=args.alpha, gamma=args.gamma,
            eps_schedule=args.eps_schedule, q_init=args.q_init,
            eval_every=args.eval_every, eval_episodes=args.eval_episodes,
            map_name=args.map_name, is_slippery=slippery, log=log,
        )
        # A final evaluation regardless of whether the periodic schedule
        # happened to land on the last episode. Reporting a score measured
        # 1,999 episodes before the end of training is a small lie that is
        # entirely avoidable.
        final = evaluate_greedy(
            result.Q, episodes=args.eval_episodes, seed=50_000 + seed,
            map_name=args.map_name, is_slippery=slippery,
        )
        runs.append((result, final))
        print(f"  seed {seed}: greedy {final['mean_return']:.3f} "
              f"± {final['stderr_return']:.3f} over {final['episodes']} episodes")

    ql_means = [f["mean_return"] for _, f in runs]
    ql_mean, ql_std, ql_stderr = mean_and_stderr(ql_means)

    # -- 3 & 4. export and register ----------------------------------------
    print("\n[3/4] exporting the best seed's Q-table")
    best_index = int(np.argmax(ql_means))
    best_result, best_final = runs[best_index]
    row = export_qtable(best_result.Q, policy_dir / f"{TRAINED_NAME}.npz")
    print(f"  {row['name']}.npz · {row['bytes']:,} bytes · sha256 {row['sha256'][:16]}… "
          f"(seed {best_result.seed}, greedy {best_final['mean_return']:.3f})")

    print("\n[4/4] registering artifacts")
    try:
        register(row, experiment_id=best_result.experiment_id)
    except Exception as exc:  # noqa: BLE001
        print(f"  artifact written but not registered: {exc}")
    untrained = export_untrained_baseline(policy_dir, map_name=args.map_name)
    print(f"  {untrained['name']}.npz · {untrained['bytes']:,} bytes "
          f"(the 'random' side of the Watch tab)")

    if untrained["sha256"] == row["sha256"]:
        # Reachable, and worth naming rather than leaving as a mystery: if the
        # run never once reached the goal, Q is still all zeros, so the trained
        # artifact is byte-identical to the untrained one. Identical bytes mean
        # an identical checksum — GET /policies de-duplicates by sha256 and the
        # `policies` table has a UNIQUE constraint on it — so you will see one
        # artifact where you expected two, and the registry insert above will
        # have been rejected. Both are symptoms of a run that did not learn,
        # not of a second bug.
        print(
            "  NOTE: the exported table is byte-identical to the untrained one, "
            "which means Q never moved. Train for longer before you go looking "
            "for a serving bug."
        )

    summary = {
        "env_id": f"{ENV_ID}-{args.map_name}-{'slippery' if slippery else 'deterministic'}",
        "seeds": args.seeds,
        "episodes_per_seed": args.episodes,
        "hyperparameters": {
            "alpha": args.alpha, "gamma": args.gamma,
            "eps_schedule": args.eps_schedule, "q_init": args.q_init,
        },
        "random_mean_return": rnd_mean,
        "random_stderr": rnd_stderr,
        "qlearning_per_seed": ql_means,
        "qlearning_mean_return": ql_mean,
        "qlearning_across_seed_std": ql_std,
        "qlearning_across_seed_stderr": ql_stderr,
        "exported": row | {"seed": best_result.seed,
                           "greedy_mean_return": best_final["mean_return"]},
        "wall_clock_seconds": round(time.time() - started, 1),
    }

    print("\n" + "=" * 78)
    print(f"{'configuration':<24}{'seeds':>7}{'mean return':>14}{'± SE':>10}")
    print("-" * 78)
    print(f"{'random (baseline)':<24}{args.seeds:>7}{rnd_mean:>14.4f}{rnd_stderr:>10.4f}")
    print(f"{'q-learning (greedy)':<24}{args.seeds:>7}{ql_mean:>14.4f}{ql_stderr:>10.4f}")
    print("-" * 78)
    print(f"per-seed greedy returns: {[round(m, 3) for m in ql_means]}")
    print(f"exported: policies/{row['name']}.npz (seed {best_result.seed}) — "
          f"the BEST seed, which is not the number to report")
    print(f"report:   {ql_mean:.3f} ± {ql_stderr:.3f} across {args.seeds} seeds")
    if ql_mean <= rnd_mean:
        print(
            "\nWARNING: the learner did not beat the baseline. On the slippery 8x8 "
            "lake that is the expected outcome of too few episodes — the goal is "
            "reached by chance perhaps once in a hundred episodes, so there is "
            "little to learn from until enough of those have accumulated. Check "
            "the greedy evaluations in `evaluations` before changing alpha."
        )
    print("=" * 78)
    print(json.dumps(summary, indent=2, default=str))

    # Restarting the service picks the artifact up; POST /reload does it live.
    print("\nNext:  uvicorn api.main:app --reload --port 8000   (then POST /reload)")
    return summary


if __name__ == "__main__":
    main()
