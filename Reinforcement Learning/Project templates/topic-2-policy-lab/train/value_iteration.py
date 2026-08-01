"""
train/value_iteration.py — the PLANNER. It is handed the model and computes the
optimal policy exactly.

    python -m train.value_iteration

What makes this file a planner and not a learner is one line of its input: it
reads `env.unwrapped.P` and never calls `env.step()`. It therefore has no
seed, no exploration parameter, no variance, and no learning curve. It has a
CONVERGENCE curve instead, which is a different object measuring a different
thing, and conflating the two is the mistake this product exists to make
visible.

The Bellman optimality backup, once, in words: the value of a state under the
best policy is the value of the best action available in it, and the value of
an action is the immediate reward plus the discounted value of wherever it
lands you, averaged over where it might land you.

    Q_{k+1}(s, a) = sum_{s'} P(s'|s,a) [ R(s,a,s') + gamma * V_k(s') ]
    V_{k+1}(s)    = max_a Q_{k+1}(s, a)

Why the Bellman residual is logged per sweep
---------------------------------------------
`max_s |V_{k+1}(s) - V_k(s)|` is the only stopping criterion here that comes
with a guarantee. The backup operator is a gamma-contraction in the max norm,
so a residual below theta bounds the remaining error:

    ||V_k - V*||_inf  <=  theta * gamma / (1 - gamma)

which means "I stopped because the numbers stopped moving" becomes "I stopped
because my answer is provably within 0.00019 of exact." That sentence is worth
a row in the database, so every sweep writes one — with the residual in its own
column, because the `return` column already means something else and
overloading it is how a schema becomes unreadable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from envs import ACTION_ARROWS, ENV_ID, GridSpec, make_env
from shared.config import get_settings
from shared.preprocess import dense_model
from shared.store import get_store
from train.export import export_tabular_policy, register

# The name the artifact is registered under, and the string `POST /act` maps
# `policy_source="value_iteration"` onto. Defined here rather than typed as a
# literal in three files: a rename that misses one of them produces a 404 at
# runtime and nothing at all at import time.
ARTIFACT_NAME = "value_iteration"

DEFAULT_GAMMA = 0.95
# Convergence tolerance on the max-norm residual. 1e-10 rather than something
# like 1e-4 because value iteration on 25 states costs microseconds per sweep:
# the exact solution is the reference every Monte Carlo number in this product
# is measured against, and buying four more digits of it costs ~40 more sweeps.
DEFAULT_THETA = 1e-10


@dataclass
class PlanResult:
    """Everything the planner produces. Kept in one object so that the training
    entry point, the tests and the compare script cannot disagree about what
    "the exact solution" means."""

    V: np.ndarray  # (n_states,)   optimal state values
    Q: np.ndarray  # (n_states, n_actions) optimal action values
    policy: np.ndarray  # (n_states,) greedy action per state
    residuals: list[float] = field(default_factory=list)
    gamma: float = DEFAULT_GAMMA

    @property
    def sweeps(self) -> int:
        return len(self.residuals)

    @property
    def error_bound(self) -> float:
        """The guaranteed max-norm distance from V* implied by the last residual."""
        if not self.residuals:
            return float("inf")
        return self.residuals[-1] * self.gamma / (1.0 - self.gamma)


def value_iteration(
    P,
    n_states: int,
    n_actions: int,
    gamma: float = DEFAULT_GAMMA,
    theta: float = DEFAULT_THETA,
    max_sweeps: int = 10_000,
    on_sweep: Callable[[int, np.ndarray, float], None] | None = None,
) -> PlanResult:
    """Bellman optimality backups until the residual falls below theta.

    Takes `P` in Gymnasium's convention rather than an environment object, so
    the same function plans for FrozenLake, for the 2x2 grid the unit test
    builds by hand, and for anything else with a tabulated model. A planner
    that takes an env is a planner you cannot test on a case you can solve on
    paper.

    `on_sweep(k, V_k, residual)` is called once per sweep. Telemetry enters
    through a callback rather than by the function writing rows itself, because
    the moment this function knows about the store it can no longer be called
    from a unit test without a database, and Topic 5 cannot reuse it at all.
    """
    if not 0.0 <= gamma < 1.0:
        # gamma = 1 is legal for a guaranteed-terminating episodic task but the
        # contraction argument above — and therefore the error bound and the
        # stopping rule — evaporate. Refuse it here rather than return a number
        # whose accuracy nobody can state.
        raise ValueError(f"gamma must be in [0, 1) for the contraction bound to hold, got {gamma}")

    T, R, B = dense_model(P, n_states, n_actions)
    # Precombine: TB[s, a, s'] is the probability mass that is allowed to
    # propagate VALUE backwards, which is zero across a terminal transition.
    TB = T * B

    V = np.zeros(n_states, dtype=np.float64)
    residuals: list[float] = []
    Q = np.zeros((n_states, n_actions), dtype=np.float64)

    for k in range(max_sweeps):
        # One synchronous sweep. "Synchronous" means every state is backed up
        # from the SAME V_k. In-place (Gauss-Seidel) updates converge faster in
        # practice but make the residual depend on the order you happened to
        # iterate the states in, which makes the convergence curve in the UI
        # unreproducible across implementations.
        Q = R + gamma * TB @ V
        V_next = Q.max(axis=1)
        residual = float(np.max(np.abs(V_next - V)))
        V = V_next
        residuals.append(residual)
        if on_sweep is not None:
            on_sweep(k, V, residual)
        if residual < theta:
            break

    # np.argmax breaks ties toward the lowest action index. That is a real
    # decision, not a detail: on a symmetric grid several actions are genuinely
    # optimal, and a policy that resolved ties randomly would produce a
    # different artifact checksum on every run and make the audit trail useless.
    policy = np.argmax(Q, axis=1).astype(np.int64)
    return PlanResult(V=V, Q=Q, policy=policy, residuals=residuals, gamma=gamma)


def policy_evaluation(
    P,
    policy: np.ndarray,
    n_states: int,
    n_actions: int,
    gamma: float = DEFAULT_GAMMA,
) -> np.ndarray:
    """v_pi for a FIXED deterministic policy, solved exactly as a linear system.

    Used by `train/compare.py` as the ground truth the Monte Carlo estimate is
    scored against. It is a direct solve of (I - gamma * P_pi) v = r_pi rather
    than iterated backups because at 25 states the solve is exact to machine
    precision, and a "ground truth" carrying its own iteration error would make
    a small RMSE impossible to distinguish from a small reference error.

    For the greedy policy returned by `value_iteration` this reproduces its V
    to ~1e-15, which `tests/test_topic2.py` asserts — the two routes to the
    same number are each other's check.
    """
    T, R, B = dense_model(P, n_states, n_actions)
    rows = np.arange(n_states)
    P_pi = (T * B)[rows, policy]  # (n_states, n_states)
    r_pi = R[rows, policy]
    return np.linalg.solve(np.eye(n_states) - gamma * P_pi, r_pi)


def render_policy(policy: np.ndarray, rows: int, cols: int, terminal: set[int]) -> str:
    """The greedy policy as an arrow grid. Print it — a value function is hard
    to eyeball for correctness, an arrow map is not."""
    out = []
    for r in range(rows):
        line = []
        for c in range(cols):
            s = r * cols + c
            line.append("*" if s in terminal else ACTION_ARROWS[int(policy[s])])
        out.append(" ".join(line))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Entry point: plan, log every sweep, persist the artifact.
# ---------------------------------------------------------------------------


def run(gamma: float = DEFAULT_GAMMA, theta: float = DEFAULT_THETA,
        quiet: bool = False) -> tuple[PlanResult, dict[str, Any]]:
    env = make_env(time_limit=False)
    core = env.unwrapped
    spec: GridSpec = core.spec_
    store = get_store()
    experiment_id = store.insert_experiment(
        {
            "algorithm": "value_iteration",
            "env_id": ENV_ID,
            # A planner has no seed. Logging 0 rather than omitting the column
            # keeps the row shape identical to a learner's, so the run-history
            # view needs no special case — and the README says plainly that the
            # value is a placeholder, because a "seed" that means nothing is
            # otherwise read as a seed that means something.
            "seed": 0,
            "hyperparameters": {
                "gamma": gamma,
                "theta": theta,
                "slip": spec.slip,
                "reward": spec.reward.as_dict(),
                "seed_is_meaningless": True,
            },
            "git_sha": get_settings().git_sha,
            "notes": "exact solution; reads env.unwrapped.P and never calls step()",
        }
    )

    # One row per sweep, collected by the callback. `return` carries V(start
    # state) so the run-history view shows something a stakeholder recognises
    # ("what does the plan think the job is worth from the depot?"), `length`
    # carries the number of backups the sweep performed, and the residual gets
    # its own column — added by db/migrations/002_topic2.sql, because the
    # `return` column already means something else and overloading a column is
    # how a schema stops being self-describing.
    sweep_rows: list[dict[str, Any]] = []

    def log_sweep(k: int, V_k: np.ndarray, residual: float) -> None:
        sweep_rows.append(
            {
                "experiment_id": experiment_id,
                "episode_index": k,
                "return": float(V_k[core.start_state]),
                "length": core.n_states * core.n_actions,
                "epsilon": None,  # a planner does not explore. Null, not 0.0.
                "bellman_residual": float(residual),
            }
        )

    plan = value_iteration(
        core.P, core.n_states, core.n_actions, gamma=gamma, theta=theta, on_sweep=log_sweep
    )
    store.insert_episodes(sweep_rows)

    row = export_tabular_policy(
        Q=plan.Q,
        V=plan.V,
        policy=plan.policy,
        path=f"{get_settings().policy_dir}/{ARTIFACT_NAME}.npz",
    )
    register(row, experiment_id=experiment_id)

    if not quiet:
        print(f"value iteration: {plan.sweeps} sweeps, "
              f"final residual {plan.residuals[-1]:.3e}, "
              f"error bound {plan.error_bound:.3e}")
        print(f"V(start) = {plan.V[core.start_state]:.6f}")
        print(render_policy(plan.policy, core.rows, core.cols, set(core.terminal_states)))
        print(json.dumps({"experiment_id": experiment_id, **row}, indent=2))
    return plan, row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--theta", type=float, default=DEFAULT_THETA)
    args = parser.parse_args()
    run(gamma=args.gamma, theta=args.theta)


if __name__ == "__main__":
    main()
