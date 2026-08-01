"""
train/random_agent.py — the baseline. Build step 4.

    python -m train.random_agent                      # 1,000 episodes x 3 seeds
    python -m train.random_agent --episodes 1000 --seeds 5

A random policy is not a warm-up exercise, it is the number every claim in your
report is measured against. "The agent reaches 0.55" means nothing until the
reader knows what falling over reaches. On the slippery 8x8 lake a uniform
random policy reaches the goal on the order of one episode in a hundred, and
producing that number yourself — with a standard error attached — is what turns
your trained result from a score into a comparison.

Two things this file does that are easy to skip and expensive to skip:

  * It writes ONE ROW PER EPISODE to `episodes`, exactly as the learner does.
    The baseline has to be queryable from the same table under the same schema,
    or the comparison in your report is between a database and a screenshot.

  * It logs `epsilon = 1.0`. A uniform random policy IS epsilon-greedy with
    epsilon pinned at one, and recording it that way means the baseline and the
    learner sit on one axis instead of being two unrelated things you eyeball.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from envs import ENV_ID, MAP_NAME, make_env
from train.telemetry import (
    EpisodeBuffer,
    mean_and_stderr,
    new_experiment,
    warn_if_data_tier_is_local,
)


def run_random_agent(
    episodes: int = 1000,
    seed: int = 0,
    map_name: str = MAP_NAME,
    is_slippery: bool = True,
    log: bool = True,
) -> dict:
    """Run a uniform random policy and return its per-episode returns.

    Note the two independent sources of randomness and why they are separate:
    `rng` chooses actions, and the integers it draws seed the ENVIRONMENT's
    transitions. Seeding the environment once at the start and never again
    would make every episode replay the same slips, which would understate the
    variance the baseline exists to measure.
    """
    env = make_env(map_name=map_name, is_slippery=is_slippery)
    n_actions = int(env.action_space.n)
    rng = np.random.default_rng(seed)

    experiment_id = None
    if log:
        experiment_id = new_experiment(
            algorithm="random",
            env_id=f"{ENV_ID}-{map_name}-{'slippery' if is_slippery else 'deterministic'}",
            seed=seed,
            hyperparameters={"policy": "uniform", "episodes": episodes},
        )

    returns: list[float] = []
    lengths: list[int] = []
    buffer = EpisodeBuffer(experiment_id) if experiment_id else None

    try:
        for ep in range(episodes):
            # A fresh environment seed per episode, drawn from the run's own
            # generator: reproducible from `seed` alone, and different every
            # episode. `env.reset()` with no seed would also vary, but nothing
            # you wrote down would let anyone reproduce it.
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            total, steps, done = 0.0, 0, False
            while not done:
                action = int(rng.integers(n_actions))
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                steps += 1
                done = terminated or truncated
            returns.append(total)
            lengths.append(steps)
            if buffer:
                buffer.add(episode_index=ep, ret=total, length=steps, epsilon=1.0)
    finally:
        # `finally`, so that interrupting a long baseline still leaves the
        # episodes it completed in the table rather than in a dead process.
        if buffer:
            buffer.flush()
        env.close()

    mean, std, stderr = mean_and_stderr(returns)
    return {
        "experiment_id": experiment_id,
        "seed": seed,
        "episodes": episodes,
        "returns": returns,
        "mean_return": mean,
        "std_return": std,
        "stderr_return": stderr,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
    }


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Random-policy baseline on the Lake Pilot environment.")
    p.add_argument("--episodes", type=int, default=1000,
                   help="episodes per seed (build step 4 asks for 1,000)")
    p.add_argument("--seeds", type=int, default=3,
                   help="number of independent seeds; the syllabus floor is 3")
    p.add_argument("--map", dest="map_name", default=MAP_NAME)
    p.add_argument("--no-slip", action="store_true",
                   help="deterministic lake — for debugging only, never for a reported number")
    p.add_argument("--no-log", action="store_true", help="skip the data tier entirely")
    args = p.parse_args(argv)

    if not args.no_log:
        warn_if_data_tier_is_local()

    per_seed = []
    for seed in range(args.seeds):
        r = run_random_agent(
            episodes=args.episodes,
            seed=seed,
            map_name=args.map_name,
            is_slippery=not args.no_slip,
            log=not args.no_log,
        )
        per_seed.append(r)
        print(
            f"seed {seed}: mean return {r['mean_return']:.4f} "
            f"± {r['stderr_return']:.4f} (SE, n={args.episodes})  "
            f"mean length {r['mean_length']:.1f}"
        )

    # Pooling across seeds rather than averaging the per-seed means: every
    # episode is one observation of the same policy, so the pooled standard
    # error is over all of them. Averaging the means would quietly discard the
    # within-seed spread and make the baseline look more certain than it is.
    pooled = [x for r in per_seed for x in r["returns"]]
    mean, std, stderr = mean_and_stderr(pooled)
    summary = {
        "algorithm": "random",
        "seeds": args.seeds,
        "episodes_per_seed": args.episodes,
        "episodes_total": len(pooled),
        "mean_return": mean,
        "std_return": std,
        "stderr_return": stderr,
    }
    print(json.dumps(summary, indent=2))
    print(
        f"\nBaseline: {mean:.4f} ± {stderr:.4f} over {len(pooled)} episodes "
        f"and {args.seeds} seeds. Quote it with the standard error and the "
        f"seed count, every time."
    )
    return summary


if __name__ == "__main__":
    main()
