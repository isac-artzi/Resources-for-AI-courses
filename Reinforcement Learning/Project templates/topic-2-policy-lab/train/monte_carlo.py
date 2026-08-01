"""
train/monte_carlo.py — the LEARNER. It never sees the model.

    python -m train.monte_carlo

Two algorithms live here, and they answer different questions:

  * `first_visit_mc_evaluation` — PREDICTION. Given a policy, what is it worth?
    Average the observed return from the first visit to each state. This is the
    estimator whose distance from the exact solution `train/compare.py`
    measures.

  * `mc_control_exploring_starts` — CONTROL. Find a good policy. Generalised
    policy iteration where the evaluation step is a sampled average and the
    improvement step is a greedy argmax over Q.

The only thing either of them is allowed to touch is `env.reset()` and
`env.step()`. If you find yourself reaching for `env.unwrapped.P` in this file,
stop: the entire claim of this product is that the learner does not have the
model, and a learner that peeks is not evidence of anything.

Why exploring starts, stated plainly
-------------------------------------
Monte Carlo estimates the value of a state by averaging returns observed FROM
that state. A state that is never visited gets no estimate at all. Under the
optimal policy from a fixed depot, this grid's agent walks essentially one
corridor and touches maybe eight of the twenty-five cells — so an RMSE computed
over all states would be dominated by cells with zero samples, and would
measure the geometry of the grid rather than the quality of the estimator.

Exploring starts fixes that by beginning each episode in a uniformly chosen
state (and, for control, a uniformly chosen first action). It is an assumption,
not a free lunch: it requires an environment you can reset into any state,
which a simulator gives you and a real forklift does not. The alternative for a
deployed system is a soft policy — epsilon-greedy — which keeps every action
reachable at the cost of evaluating a policy slightly different from the one
you intend to ship. Both are in the Concepts tab; this file implements the
first, and says so in the model card.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from envs import ENV_ID, make_env
from shared.config import get_settings
from shared.preprocess import discounted_returns, first_visit_indices
from shared.store import get_store
from train.export import export_tabular_policy
from train.export import register as register_artifact
from train.value_iteration import DEFAULT_GAMMA

ARTIFACT_NAME = "monte_carlo"

# Default training budget for the control agent: 100k episodes, about six
# seconds on a laptop.
#
# Measured, and worth knowing before you spend an evening tuning: on this grid
# the greedy policy is IDENTICAL at 20k, 50k, 100k and 200k episodes, and worth
# 0.3093 at the depot against the optimal 0.3176 — a 2.6% shortfall that a
# tenfold budget increase does not touch. The two cells responsible differ by
# 0.006 in Q*, and exploring starts hands each state-action pair only about
# 1/100th of the episodes as a first visit, so at 100k episodes those pairs have
# ~1,100 samples of a return whose spread is an order of magnitude larger than
# the difference being resolved. Control plateaus for a reason you can compute,
# not because the implementation is broken. See the README.
#
# The convergence STUDY in train/compare.py uses its own, smaller budgets:
# its question is how fast the PREDICTION arrives, not where control stops.
DEFAULT_CONTROL_EPISODES = 100_000
DEFAULT_EVAL_EPISODES = 10_000


@dataclass
class Episode:
    """One trajectory, in the only form Monte Carlo can use: complete.

    States, actions and rewards are parallel lists with `rewards[t]` being the
    reward received on the transition OUT of `states[t]`. Getting that offset
    wrong by one is the single most common Monte Carlo bug and it does not
    crash — it produces a value function that is wrong by exactly one discount
    factor and looks entirely reasonable.
    """

    states: list[int] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    truncated: bool = False

    def __len__(self) -> int:
        return len(self.states)


def generate_episode(
    env,
    policy: np.ndarray,
    rng: np.random.Generator,
    start_state: int | None = None,
    start_action: int | None = None,
    max_steps: int = 1000,
) -> Episode:
    """Roll out one episode under a deterministic tabular policy.

    `start_state` / `start_action` implement exploring starts. `max_steps` is a
    second belt on top of the environment's TimeLimit: a bug that produces a
    policy which cycles forever would otherwise hang a training run with no
    output, and a hung run at 2 a.m. is indistinguishable from a slow one.
    """
    options = None if start_state is None else {"state": int(start_state)}
    obs, _ = env.reset(options=options)
    ep = Episode()

    for t in range(max_steps):
        state = int(obs)
        if t == 0 and start_action is not None:
            action = int(start_action)
        else:
            action = int(policy[state])
        obs, reward, terminated, truncated, _ = env.step(action)
        ep.states.append(state)
        ep.actions.append(action)
        ep.rewards.append(float(reward))
        if terminated or truncated:
            ep.truncated = bool(truncated) and not bool(terminated)
            break
    else:
        ep.truncated = True
    return ep


@dataclass
class MCEvaluationResult:
    V: np.ndarray
    visits: np.ndarray
    snapshots: dict[int, np.ndarray] = field(default_factory=dict)
    episode_rows: list[dict[str, Any]] = field(default_factory=list)
    truncated_episodes: int = 0


def first_visit_mc_evaluation(
    env,
    policy: np.ndarray,
    episodes: int,
    gamma: float = DEFAULT_GAMMA,
    seed: int = 0,
    snapshot_at: Sequence[int] = (),
    exploring_starts: bool = True,
    every_visit: bool = False,
    collect_rows: bool = True,
) -> MCEvaluationResult:
    """First-visit Monte Carlo prediction: estimate v_pi by averaging returns.

    `every_visit=True` flips the one line that distinguishes the two
    estimators, so the Concepts tab's claim that they differ only in which
    visits count is checkable rather than asserted.

    The running mean is incremental — V[s] += (G - V[s]) / n — rather than a
    list of every return per state. Storing the returns would let you compute a
    per-state standard error, which is genuinely useful, but at 30,000 episodes
    it is also a few hundred thousand floats held in the training process for
    a number that `train/compare.py` gets more honestly from independent seeds.
    Across-seed variation includes the estimator's own dependence on its random
    stream; within-run variation does not.
    """
    core = env.unwrapped
    n_states = core.n_states
    non_terminal = np.array(
        [s for s in range(n_states) if s not in core.terminal_states], dtype=np.int64
    )

    V = np.zeros(n_states, dtype=np.float64)
    visits = np.zeros(n_states, dtype=np.int64)

    # One generator for the whole run, and the environment seeded exactly once.
    # Re-seeding the environment every episode with seed + episode_index is a
    # common and subtly wrong habit: consecutive seeds produce correlated
    # streams, so the "independent" episodes are not, and the confidence
    # intervals computed from them are too narrow.
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)

    result = MCEvaluationResult(V=V, visits=visits)
    wanted = sorted(set(int(b) for b in snapshot_at))

    for k in range(episodes):
        start = int(rng.choice(non_terminal)) if exploring_starts else None
        ep = generate_episode(env, policy, rng, start_state=start)
        G = discounted_returns(ep.rewards, gamma)

        if every_visit:
            steps = range(len(ep))
        else:
            steps = first_visit_indices(ep.states).values()

        for t in steps:
            s = ep.states[t]
            visits[s] += 1
            V[s] += (G[t] - V[s]) / visits[s]

        if ep.truncated:
            result.truncated_episodes += 1
        if collect_rows:
            result.episode_rows.append(
                {
                    "episode_index": k,
                    "return": float(G[0]) if len(G) else 0.0,
                    "length": len(ep),
                    # Exploring starts is not epsilon-greedy. There is no
                    # exploration parameter in force, so the column is NULL —
                    # writing 0.0 would read as "greedy with no exploration",
                    # which is the opposite of what this run did.
                    "epsilon": None,
                }
            )
        if wanted and k + 1 == wanted[0]:
            result.snapshots[wanted.pop(0)] = V.copy()

    return result


@dataclass
class MCControlResult:
    Q: np.ndarray
    policy: np.ndarray
    visits: np.ndarray
    episode_rows: list[dict[str, Any]] = field(default_factory=list)
    truncated_episodes: int = 0

    @property
    def V(self) -> np.ndarray:
        """The control agent's value estimate: max_a Q(s, a).

        Note that this is NOT the same object as the first-visit evaluation of
        the final policy. It is an average of returns collected while the
        policy was still changing, so early, worse-policy returns are baked
        into it. It is reported in the Value Map tab labelled as what it is.
        """
        return self.Q.max(axis=1)


def mc_control_exploring_starts(
    env,
    episodes: int = DEFAULT_CONTROL_EPISODES,
    gamma: float = DEFAULT_GAMMA,
    seed: int = 0,
    collect_rows: bool = True,
) -> MCControlResult:
    """Monte Carlo control with exploring starts (Sutton & Barto's ES variant).

    Generalised policy iteration with both halves approximated:

        evaluate   Q(s, a) <- running mean of first-visit returns
        improve    pi(s)   <- argmax_a Q(s, a)

    The improvement step runs after EVERY episode rather than after Q has
    converged for the current policy. That is deliberate and it is what makes
    the algorithm practical: waiting for exact evaluation would mean thousands
    of episodes between policy updates. The policy improvement theorem still
    guarantees the greedy policy is no worse than the current one at every step
    — it says nothing about how accurate Q has to be first, which is why this
    works and also why it can thrash on a noisier problem than this one.

    Q is initialised to zeros. On this grid every reward is negative except at
    the goal, so zeros are OPTIMISTIC and untried actions look attractive —
    free exploration, on top of the exploring starts. Initialise Q to a large
    negative number instead and the agent becomes pessimistic and sticks to
    whatever it tried first. Neither is neutral; state which you chose.
    """
    core = env.unwrapped
    n_states, n_actions = core.n_states, core.n_actions
    non_terminal = np.array(
        [s for s in range(n_states) if s not in core.terminal_states], dtype=np.int64
    )

    Q = np.zeros((n_states, n_actions), dtype=np.float64)
    counts = np.zeros((n_states, n_actions), dtype=np.int64)
    policy = np.zeros(n_states, dtype=np.int64)

    rng = np.random.default_rng(seed)
    env.reset(seed=seed)

    result = MCControlResult(Q=Q, policy=policy, visits=counts)

    for k in range(episodes):
        s0 = int(rng.choice(non_terminal))
        a0 = int(rng.integers(n_actions))
        ep = generate_episode(env, policy, rng, start_state=s0, start_action=a0)
        G = discounted_returns(ep.rewards, gamma)

        # First visit to each (state, action) PAIR — not to each state. Control
        # needs one estimate per pair, and collapsing to first-visit-per-state
        # would silently discard the sample for every action the episode tried
        # after its first one in that cell.
        seen: set[tuple[int, int]] = set()
        for t in range(len(ep)):
            pair = (ep.states[t], ep.actions[t])
            if pair in seen:
                continue
            seen.add(pair)
            s, a = pair
            counts[s, a] += 1
            Q[s, a] += (G[t] - Q[s, a]) / counts[s, a]
            policy[s] = int(np.argmax(Q[s]))

        if ep.truncated:
            result.truncated_episodes += 1
        if collect_rows:
            result.episode_rows.append(
                {
                    "episode_index": k,
                    "return": float(G[0]) if len(G) else 0.0,
                    "length": len(ep),
                    "epsilon": None,  # exploring starts, not epsilon-greedy
                }
            )
    return result


# ---------------------------------------------------------------------------
# Entry point: learn a policy, log every episode, persist the artifact.
# ---------------------------------------------------------------------------


def run(
    control_episodes: int = DEFAULT_CONTROL_EPISODES,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    gamma: float = DEFAULT_GAMMA,
    seed: int = 0,
    quiet: bool = False,
) -> tuple[MCControlResult, MCEvaluationResult, dict[str, Any]]:
    env = make_env()
    core = env.unwrapped
    store = get_store()
    settings = get_settings()

    control = mc_control_exploring_starts(env, episodes=control_episodes, gamma=gamma, seed=seed)

    # Evaluate the LEARNED policy with a fresh, independent run. Reusing the
    # control agent's own Q as the value estimate would report the number the
    # agent optimised, which is optimistically biased by exactly the maximum it
    # was taking — the tabular ancestor of the overestimation bias that Double
    # Q-learning exists to fix.
    evaluation = first_visit_mc_evaluation(
        env, control.policy, episodes=eval_episodes, gamma=gamma, seed=seed + 10_000
    )

    # TWO experiment rows, not one. Control and prediction are different
    # algorithms answering different questions, and their episodes are not
    # comparable: a control episode's return was collected under a policy that
    # was still changing. Filing them under one experiment_id would make the
    # learning curve in the run-history view a splice of two different runs.
    common = {
        "env_id": ENV_ID,
        "seed": seed,
        "git_sha": settings.git_sha,
    }
    control_id = store.insert_experiment(
        {
            **common,
            "algorithm": "monte_carlo_es",
            "hyperparameters": {
                "gamma": gamma,
                "episodes": control_episodes,
                "exploration": "exploring_starts",
                "first_visit": True,
                "first_visit_unit": "state_action_pair",
                "return_logged": "discounted",
            },
            "notes": "model-free control; only reset() and step() were called",
        }
    )
    store.insert_episodes(
        [{"experiment_id": control_id, **row} for row in control.episode_rows]
    )

    evaluation_id = store.insert_experiment(
        {
            **common,
            "algorithm": "mc_evaluation",
            "seed": seed + 10_000,
            "hyperparameters": {
                "gamma": gamma,
                "episodes": eval_episodes,
                "exploration": "exploring_starts",
                "first_visit": True,
                "first_visit_unit": "state",
                "policy": "monte_carlo_es_greedy",
                "return_logged": "discounted",
            },
            "notes": "independent evaluation of the learned policy",
        }
    )
    store.insert_episodes(
        [{"experiment_id": evaluation_id, **row} for row in evaluation.episode_rows]
    )

    row = export_tabular_policy(
        Q=control.Q,
        # The value function shipped with this artifact is the INDEPENDENT
        # first-visit evaluation, not control's max_a Q. The Value Map tab
        # compares it against the planner's V*, and comparing an optimised
        # maximum against an exact value would put the estimator's bias into a
        # chart labelled "difference".
        V=evaluation.V,
        policy=control.policy,
        path=f"{settings.policy_dir}/{ARTIFACT_NAME}.npz",
    )
    # Registered against the CONTROL experiment: that is the run that produced
    # the policy being served. The evaluation run measured it; it did not make
    # it, and attributing an artifact to the run that scored it is how a
    # provenance chain quietly becomes wrong.
    register_artifact(row, experiment_id=control_id)

    if not quiet:
        trunc = control.truncated_episodes / max(control_episodes, 1)
        print(
            f"monte carlo (seed {seed}): {control_episodes} control episodes, "
            f"{eval_episodes} evaluation episodes"
        )
        print(f"  truncation rate during control: {trunc:.4%} "
              f"(a high rate means the TimeLimit is biasing your returns)")
        print(f"  V_hat(start) = {evaluation.V[core.start_state]:.6f}")
        # Reported over NON-TERMINAL states only. A terminal state is never
        # appended to a trajectory (the episode ends on entering it), so its
        # visit count is legitimately zero and including it here would make the
        # coverage check report 0 forever and stop being read.
        interior = [s for s in range(core.n_states) if s not in core.terminal_states]
        print(f"  least-visited non-terminal state saw "
              f"{int(evaluation.visits[interior].min())} first visits")
        print(json.dumps({"control_experiment_id": control_id,
                          "evaluation_experiment_id": evaluation_id, **row}, indent=2))
    return control, evaluation, row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-episodes", type=int, default=DEFAULT_CONTROL_EPISODES)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(
        control_episodes=args.control_episodes,
        eval_episodes=args.eval_episodes,
        gamma=args.gamma,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
