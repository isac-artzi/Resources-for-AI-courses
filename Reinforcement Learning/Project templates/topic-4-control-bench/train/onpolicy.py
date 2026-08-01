"""
train/onpolicy.py — the data collection A2C and PPO share.

TRAINING TIER (imports torch). Never reachable from api/ or ui/.

Both on-policy algorithms in this product interact with the environment in
exactly the same way — run the current policy for a fixed number of STEPS,
remember what happened, bootstrap the value of wherever you stopped — and differ
only in what they do with the result. Sharing the collector is not tidiness; it
is what makes the matched-budget comparison in `train/compare.py` mean anything.
If A2C collected complete episodes and PPO collected fixed-length segments, a
difference in their learning curves would partly be a difference in how their
batches were assembled, and no amount of seeding would separate the two.

Why a fixed number of STEPS rather than a fixed number of EPISODES
------------------------------------------------------------------
Topic 3 collected whole episodes, because a Monte Carlo return needs the episode
to have ended. With a critic we no longer need that: the value of the state we
stopped in is an estimate of everything that would have followed. Collecting by
steps is what makes "matched environment-step budget" a thing you can actually
hold fixed — an A2C run and a PPO run that have each consumed 30,000 steps have
had the same amount of experience, which is the only comparison that answers
"which is more sample-efficient".

It also matters for Acrobot specifically. An untrained Acrobot policy takes 500
steps to finish one episode, so a batch of ten episodes is 5,000 steps of a
policy that is not learning yet — and the first update would arrive after 5,000
steps rather than after 1,024.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from envs import make_env, spec_for


@dataclass
class Batch:
    """One collected segment, in the shapes the updates want it.

    `obs`, `actions`, `log_probs` and the rest are flat arrays of length
    `n_steps` — the segment may span several episodes and may cut one in half,
    which is normal and is what `truncateds` and `last_value` exist to handle.
    """

    obs: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray
    values: np.ndarray
    dones: np.ndarray          # TERMINATED — the bootstrap is zero here
    truncateds: np.ndarray     # time limit — the bootstrap is V(s'), not zero
    last_value: float
    finished_episodes: list[tuple[float, int]] = field(default_factory=list)


class StepCollector:
    """Owns one environment and the position in it, across many batches.

    Stateful on purpose. A collector that reset the environment at the start of
    every batch would throw away the second half of every episode longer than
    the batch, which on Acrobot is every episode. The environment is reset only
    when it actually ends, and the batch boundary is invisible to it.
    """

    def __init__(self, env_id: str, seed: int, max_steps: int | None = None) -> None:
        self.spec = spec_for(env_id)
        self.env_id = env_id
        self.env = make_env(env_id)
        self.max_steps = max_steps or self.spec.max_steps
        # A distinct, DERIVED seed per episode rather than one reset at the top
        # of the run. `env.reset()` with no seed uses whatever state the env
        # happens to hold, and a "seeded" run that reseeds nothing is not
        # reproducible however loudly the README says seed=0. The million-stride
        # keeps seed 0's episode seeds from ever colliding with seed 1's.
        self._seed_base = seed * 1_000_000
        self._episode_index = 0
        self.obs, _ = self.env.reset(seed=self._seed_base)
        self.env_steps = 0
        self.episode_return = 0.0
        self.episode_length = 0

    def close(self) -> None:
        self.env.close()

    def collect(self, actor, critic, n_steps: int) -> Batch:
        """Run the CURRENT policy for `n_steps` and record everything the updates need."""
        obs_buf, act_buf, logp_buf = [], [], []
        rew_buf, val_buf, done_buf, trunc_buf = [], [], [], []
        finished: list[tuple[float, int]] = []

        for _ in range(n_steps):
            obs_arr = np.asarray(self.obs, dtype=np.float32)
            action, logp = actor.act(obs_arr)
            with torch.no_grad():
                value = float(critic(torch.as_tensor(obs_arr)).item())

            next_obs, reward, terminated, truncated, _ = self.env.step(action)

            obs_buf.append(obs_arr)
            act_buf.append(action)
            logp_buf.append(logp)
            rew_buf.append(float(reward))
            val_buf.append(value)
            done_buf.append(bool(terminated))
            trunc_buf.append(bool(truncated))

            self.obs = next_obs
            self.env_steps += 1
            self.episode_return += float(reward)
            self.episode_length += 1

            if terminated or truncated:
                finished.append((self.episode_return, self.episode_length))
                self._episode_index += 1
                self.obs, _ = self.env.reset(seed=self._seed_base + self._episode_index)
                self.episode_return, self.episode_length = 0.0, 0

        # The bootstrap for the state we stopped in. Zero ONLY if that state was
        # terminal; the batch boundary itself is not a terminal state, and
        # treating it as one is the classic "my agent learns to die at step 128"
        # bug — the critic is told that everything after the batch is worth
        # nothing, which is only true if the episode really ended.
        with torch.no_grad():
            last_value = float(
                critic(torch.as_tensor(np.asarray(self.obs, dtype=np.float32))).item()
            )
        if done_buf and done_buf[-1]:
            last_value = 0.0

        return Batch(
            obs=np.asarray(obs_buf, dtype=np.float32),
            actions=np.asarray(act_buf, dtype=np.int64),
            log_probs=np.asarray(logp_buf, dtype=np.float64),
            rewards=np.asarray(rew_buf, dtype=np.float64),
            values=np.asarray(val_buf, dtype=np.float64),
            dones=np.asarray(done_buf, dtype=bool),
            truncateds=np.asarray(trunc_buf, dtype=bool),
            last_value=last_value,
            finished_episodes=finished,
        )


def evaluate_discrete(
    actor, env_id: str, episodes: int = 20, seed: int = 12345
) -> dict[str, Any]:
    """Greedy evaluation of a categorical actor. A different question from the curve.

    The training curve is what the STOCHASTIC policy scored while it was still
    exploring; this is what the deployed policy scores. Reporting the first as
    though it were the second is the most common overstatement in a first RL
    report, which is why `evaluations` is its own table.
    """
    spec = spec_for(env_id)
    env = make_env(env_id)
    returns = []
    for k in range(episodes):
        obs, _ = env.reset(seed=seed + k)
        total = 0.0
        for _ in range(spec.max_steps):
            with torch.no_grad():
                logits = actor(torch.as_tensor(np.asarray(obs, dtype=np.float32)))
            obs, reward, terminated, truncated, _ = env.step(int(torch.argmax(logits).item()))
            total += float(reward)
            if terminated or truncated:
                break
        returns.append(total)
    env.close()
    arr = np.asarray(returns, dtype=np.float64)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return {
        "episodes": int(episodes),
        "mean_return": float(arr.mean()),
        "std_return": std,
        "stderr_return": std / np.sqrt(max(len(arr), 1)),
    }


def episode_rows(
    experiment_id: str, first_index: int, finished: list[tuple[float, int]], env_steps_at_end: int
) -> list[dict[str, Any]]:
    """Turn a batch's finished episodes into `episodes` rows.

    `env_steps` is approximated as the step count at the END of the batch for
    every episode that finished inside it. The approximation is bounded by the
    batch length (1,024 steps at most here) and is stated rather than hidden,
    because the alternative — tracking the exact step index of every episode
    boundary — buys precision that no chart in this product can resolve. Say so
    if you plot at a finer resolution than the batch size.

    `epsilon` is NULL, and honestly so: none of the three algorithms has one.
    They explore through the entropy of their own policy, which is logged per
    update in `policy_updates.policy_entropy` instead.
    """
    return [
        {
            "experiment_id": experiment_id,
            # Absolute index within the run, not within the batch. The unique
            # constraint on (experiment_id, episode_index) in 001_init.sql will
            # catch a per-batch counter immediately, which is why it is there.
            "episode_index": first_index + i,
            "return": ret,
            "length": length,
            "epsilon": None,
            "env_steps": env_steps_at_end,
        }
        for i, (ret, length) in enumerate(finished)
    ]
