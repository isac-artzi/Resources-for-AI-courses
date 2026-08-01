"""
train/qlearning.py — tabular Q-learning on the Lake Pilot environment. Build step 5.

    python -m train.qlearning
    python -m train.qlearning --episodes 20000 --seeds 3 --alpha 0.1 --gamma 0.99 \
                              --eps-schedule linear:1.0:0.05:0.6

The update is one line, and it is the only line in this file that is
reinforcement learning:

    Q[s,a] <- Q[s,a] + alpha * ( r + gamma * max_a' Q[s',a'] - Q[s,a] )

Everything else here is the apparatus that makes a number produced by that line
defensible: a named epsilon schedule, one logged row per episode with the
epsilon in force at that episode, and periodic GREEDY evaluations written to a
separate table.

------------------------------------------------------------------------------
THE FOUR DECISIONS YOUR ENGINEERING REPORT HAS TO DEFEND
------------------------------------------------------------------------------
alpha (learning rate)
    How much of the new estimate replaces the old one. On a *stochastic*
    environment the target `r + gamma * max Q[s']` is noisy, so a large alpha
    makes Q chase individual lucky episodes and the greedy policy flaps between
    evaluations. The defaults here are deliberately small (0.1). If your
    learning curve is jagged rather than noisy, alpha is the first suspect.

gamma (discount)
    FrozenLake pays 1.0 at the goal and 0.0 everywhere else, so gamma is what
    tells the agent that a shorter route is worth more. It is also the only
    thing propagating value backwards from the goal across 64 states: gamma
    close to 1 propagates far but slowly, gamma small makes distant states
    indistinguishable from each other. gamma = 1.0 exactly is legal here only
    because the environment truncates at 200 steps; do not rely on that.

the epsilon schedule
    Early on, a greedy agent on this map has never seen the goal, so its argmax
    is meaningless and exploiting it wastes episodes. Late on, a high epsilon is
    a tax: every exploratory action is a step the deployed policy would not
    take, so the *training* curve sags while the *greedy* policy is fine. That
    gap is why both are logged and why they are logged separately.

the stopping criterion
    "It stopped when the loop ended" is not a criterion. Look at the greedy
    evaluations: they plateau, and everything after the plateau is compute you
    spent to make the curve longer. Say in your report which evaluation you
    stopped at and what you would have needed to see to keep going.

------------------------------------------------------------------------------
MAXIMISATION BIAS — the thing that makes this look better than it is
------------------------------------------------------------------------------
`max_a' Q[s',a']` picks the largest of four noisy estimates, and the largest of
several noisy estimates is biased upwards. On a slippery lake this means Q
systematically over-values states early in training. It does not stop
Q-learning from working, but it is why the value estimate `/act` returns is not
a promise, and it is the flaw Double Q-learning exists to fix. Worth a sentence
in your model card.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from envs import ENV_ID, MAP_NAME, make_env
from train.telemetry import (
    EpisodeBuffer,
    mean_and_stderr,
    new_experiment,
    record_evaluation,
    warn_if_data_tier_is_local,
)

# ---------------------------------------------------------------------------
# Epsilon schedules
# ---------------------------------------------------------------------------

EPS_SCHEDULE_HELP = (
    "linear:START:END:FRACTION  decay straight from START to END over the first "
    "FRACTION of the run, then hold. | "
    "exp:START:END:RATE  multiply by RATE each episode, floored at END. | "
    "const:VALUE  no decay — the control condition."
)


def parse_eps_schedule(spec: str) -> Callable[[int, int], float]:
    """Turn a schedule string into `f(episode_index, total_episodes) -> epsilon`.

    A string rather than three separate flags, for one reason: the string is
    what goes into the `hyperparameters` jsonb column verbatim. Six weeks later
    "which schedule was that run?" is a `select hyperparameters->>'eps_schedule'`
    rather than an archaeology exercise across three columns that may or may not
    have been added at the same time.

    The schedule is a function of the episode INDEX, not of wall-clock or of
    performance. Adaptive schedules that decay on improvement exist and are
    defensible, but they make a run irreproducible from its configuration
    alone — you would need the whole reward history to replay it. Start here.
    """
    parts = spec.strip().split(":")
    kind = parts[0].lower()

    if kind == "const":
        if len(parts) != 2:
            raise ValueError(f"const schedule takes one value: 'const:0.1', got {spec!r}")
        value = float(parts[1])
        return lambda ep, total: value

    if kind == "linear":
        if len(parts) != 4:
            raise ValueError(f"linear takes START:END:FRACTION, got {spec!r}")
        start, end, frac = float(parts[1]), float(parts[2]), float(parts[3])
        if not 0.0 < frac <= 1.0:
            raise ValueError("FRACTION must be in (0, 1] — it is a share of the run")

        def linear(ep: int, total: int) -> float:
            # `cut` is where decay ends. Everything after it runs at END, which
            # is the part of the run whose training return is comparable to the
            # greedy return. Reports that quote "final training return" without
            # noticing that epsilon was still 0.4 come from schedules with no
            # such flat tail.
            cut = max(1, int(total * frac))
            if ep >= cut:
                return end
            return start + (end - start) * (ep / cut)

        return linear

    if kind == "exp":
        if len(parts) != 4:
            raise ValueError(f"exp takes START:END:RATE, got {spec!r}")
        start, end, rate = float(parts[1]), float(parts[2]), float(parts[3])
        if not 0.0 < rate < 1.0:
            raise ValueError("RATE must be in (0, 1)")

        def exponential(ep: int, total: int) -> float:
            # Exponential decay is the common default and it has a trap: the
            # rate that empties in 5,000 episodes is essentially zero by 20,000,
            # so scaling a run up silently turns off exploration in the first
            # tenth of it. Print the schedule (below) and check where it lands.
            return max(end, start * (rate**ep))

        return exponential

    raise ValueError(f"unknown schedule kind {kind!r}. {EPS_SCHEDULE_HELP}")


def schedule_preview(spec: str, total: int, points: int = 5) -> list[tuple[int, float]]:
    """Sample the schedule so a run prints what its exploration will do.

    Cheap, and it catches the single most common configuration error in this
    assignment: a decay rate that reaches its floor in the first 2% of the run,
    turning a 20,000-episode experiment into a 400-episode one.
    """
    f = parse_eps_schedule(spec)
    idx = [int(total * k / (points - 1)) for k in range(points)]
    return [(i, round(f(min(i, total - 1), total), 4)) for i in idx]


# ---------------------------------------------------------------------------
# The learner
# ---------------------------------------------------------------------------


@dataclass
class QLearningResult:
    Q: np.ndarray
    returns: list[float] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    evaluations: list[dict] = field(default_factory=list)
    experiment_id: str | None = None
    seed: int = 0
    hyperparameters: dict = field(default_factory=dict)

    @property
    def final_eval(self) -> dict | None:
        return self.evaluations[-1] if self.evaluations else None


def evaluate_greedy(
    Q: np.ndarray,
    episodes: int = 100,
    seed: int = 10_000,
    map_name: str = MAP_NAME,
    is_slippery: bool = True,
) -> dict:
    """Run the greedy policy — epsilon = 0 — and return mean, std and stderr.

    This is the number the deployed service produces, and it is not the same
    number as the training curve's tail. Two details are load-bearing:

      * A FRESH environment. Evaluating inside the training environment would
        advance the same random stream the learner is drawing from, so the
        training run would come out differently depending on how often you
        chose to evaluate it.
      * A seed offset well clear of the training seeds (default 10,000). Sharing
        seeds between training and evaluation is how an agent scores well on
        exactly the slip sequences it was trained on. It is the tabular version
        of testing on your training set, and it is just as invisible.
    """
    env = make_env(map_name=map_name, is_slippery=is_slippery)
    returns: list[float] = []
    try:
        for i in range(episodes):
            obs, _ = env.reset(seed=seed + i)
            total, done = 0.0, False
            while not done:
                action = int(np.argmax(Q[int(obs)]))
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                done = terminated or truncated
            returns.append(total)
    finally:
        env.close()
    mean, std, stderr = mean_and_stderr(returns)
    return {
        "episodes": episodes,
        "mean_return": mean,
        "std_return": std,
        "stderr_return": stderr,
        "returns": returns,
    }


def q_learning(
    episodes: int = 20_000,
    seed: int = 0,
    alpha: float = 0.1,
    gamma: float = 0.99,
    eps_schedule: str = "linear:1.0:0.05:0.6",
    eval_every: int = 2_000,
    eval_episodes: int = 100,
    map_name: str = MAP_NAME,
    is_slippery: bool = True,
    log: bool = True,
    q_init: float = 0.0,
    progress: bool = True,
) -> QLearningResult:
    """Train a Q-table. Every episode is logged; evaluations are periodic."""
    env = make_env(map_name=map_name, is_slippery=is_slippery)
    n_states, n_actions = int(env.observation_space.n), int(env.action_space.n)
    eps_at = parse_eps_schedule(eps_schedule)

    # Zero initialisation is neutral, not "no opinion": on a reward-in-[0,1]
    # environment it is *pessimistic*, because every unvisited action looks
    # exactly as good as an action known to lead nowhere. Optimistic
    # initialisation (`--q-init 1.0`) makes unvisited actions look better than
    # anything yet tried and so drives systematic exploration without epsilon.
    # It is exposed as a flag because comparing the two is a good second
    # configuration for your results table.
    Q = np.full((n_states, n_actions), float(q_init), dtype=np.float64)

    rng = np.random.default_rng(seed)

    hyperparameters = {
        "alpha": alpha,
        "gamma": gamma,
        "eps_schedule": eps_schedule,
        "episodes": episodes,
        "q_init": q_init,
        "eval_every": eval_every,
        "eval_episodes": eval_episodes,
    }
    experiment_id = None
    if log:
        experiment_id = new_experiment(
            algorithm="q-learning",
            env_id=f"{ENV_ID}-{map_name}-{'slippery' if is_slippery else 'deterministic'}",
            seed=seed,
            hyperparameters=hyperparameters,
        )

    result = QLearningResult(
        Q=Q, experiment_id=experiment_id, seed=seed, hyperparameters=hyperparameters
    )
    buffer = EpisodeBuffer(experiment_id) if experiment_id else None

    try:
        for ep in range(episodes):
            epsilon = eps_at(ep, episodes)
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            state = int(obs)
            total, steps, done = 0.0, 0, False

            while not done:
                if rng.random() < epsilon:
                    action = int(rng.integers(n_actions))
                else:
                    action = int(np.argmax(Q[state]))

                obs, reward, terminated, truncated, _ = env.step(action)
                next_state = int(obs)

                # The bootstrap is dropped on TERMINATION and kept on
                # TRUNCATION, and the distinction is not pedantry. Terminated
                # means the episode really ended — the future is worth zero.
                # Truncated means the 200-step limit fired while the future was
                # still worth something; bootstrapping through it is correct,
                # and treating it as terminal teaches the agent that surviving
                # 200 steps is as bad as drowning.
                best_next = 0.0 if terminated else float(np.max(Q[next_state]))
                td_target = float(reward) + gamma * best_next
                Q[state, action] += alpha * (td_target - Q[state, action])

                state = next_state
                total += float(reward)
                steps += 1
                done = terminated or truncated

            result.returns.append(total)
            result.epsilons.append(epsilon)
            result.lengths.append(steps)
            if buffer:
                # EVERY episode, with the epsilon that was in force for it.
                # Sampling every tenth episode would halve the row count and
                # destroy the one query that separates "learning stalled" from
                # "exploration was still high".
                buffer.add(episode_index=ep, ret=total, length=steps, epsilon=epsilon)

            if eval_every and (ep + 1) % eval_every == 0:
                ev = evaluate_greedy(
                    Q,
                    episodes=eval_episodes,
                    seed=10_000 + ep,
                    map_name=map_name,
                    is_slippery=is_slippery,
                )
                ev["at_training_episode"] = ep + 1
                result.evaluations.append(ev)
                if experiment_id:
                    record_evaluation(experiment_id, ep + 1, ev["returns"])
                if progress:
                    train_tail = float(np.mean(result.returns[-eval_every:]))
                    print(
                        f"  ep {ep + 1:>7,}  eps {epsilon:5.3f}  "
                        f"train(last {eval_every}) {train_tail:6.3f}  "
                        f"greedy {ev['mean_return']:6.3f} ± {ev['stderr_return']:.3f}"
                    )
    finally:
        if buffer:
            buffer.flush()
        env.close()

    return result


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(
        description="Tabular Q-learning on the Lake Pilot environment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"epsilon schedules:\n  {EPS_SCHEDULE_HELP}",
    )
    p.add_argument("--episodes", type=int, default=20_000,
                   help="episodes per seed (the syllabus floor for this product is 20,000)")
    p.add_argument("--seeds", type=int, default=3,
                   help="number of independent seeds; the syllabus floor is 3")
    p.add_argument("--alpha", type=float, default=0.1, help="learning rate")
    p.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    p.add_argument("--eps-schedule", default="linear:1.0:0.05:0.6", help=EPS_SCHEDULE_HELP)
    p.add_argument("--q-init", type=float, default=0.0,
                   help="initial Q value; 1.0 is optimistic initialisation")
    p.add_argument("--eval-every", type=int, default=2_000,
                   help="greedy evaluation interval in episodes; 0 disables")
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--map", dest="map_name", default=MAP_NAME)
    p.add_argument("--no-slip", action="store_true",
                   help="deterministic lake — for debugging only, never for a reported number")
    p.add_argument("--no-log", action="store_true", help="skip the data tier entirely")
    args = p.parse_args(argv)

    if not args.no_log:
        warn_if_data_tier_is_local()

    print(f"epsilon schedule {args.eps_schedule!r} over {args.episodes:,} episodes:")
    for ep, eps in schedule_preview(args.eps_schedule, args.episodes):
        print(f"  episode {ep:>7,} -> eps {eps}")

    finals = []
    for seed in range(args.seeds):
        print(f"\nseed {seed}")
        r = q_learning(
            episodes=args.episodes,
            seed=seed,
            alpha=args.alpha,
            gamma=args.gamma,
            eps_schedule=args.eps_schedule,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            map_name=args.map_name,
            is_slippery=not args.no_slip,
            log=not args.no_log,
            q_init=args.q_init,
        )
        ev = r.final_eval or evaluate_greedy(
            r.Q, episodes=args.eval_episodes,
            map_name=args.map_name, is_slippery=not args.no_slip,
        )
        finals.append(ev["mean_return"])
        print(f"seed {seed}: greedy {ev['mean_return']:.3f} ± {ev['stderr_return']:.3f}")

    # Across-seed spread, reported separately from within-seed standard error.
    # They answer different questions: the standard error asks "how well do we
    # know this run's score", the across-seed spread asks "would a different
    # seed have given a different answer". On a stochastic environment the
    # second is usually the larger of the two, and it is the one a reader wants.
    mean, std, stderr = mean_and_stderr(finals)
    summary = {
        "algorithm": "q-learning",
        "seeds": args.seeds,
        "episodes_per_seed": args.episodes,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "eps_schedule": args.eps_schedule,
        "per_seed_greedy_return": finals,
        "mean_greedy_return": mean,
        "across_seed_std": std,
        "across_seed_stderr": stderr,
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
