"""
train/vpg.py — vanilla policy gradient, instrumented.

TRAINING TIER (imports torch). Never reachable from api/ or ui/.

This is the algorithm the whole product is about, and it is short. The reason
the file is long is the instrumentation: measuring the variance of the gradient
estimate costs more code than taking the gradient step does, and measuring it is
the deliverable.

The loop
--------
    1. Collect a batch of complete episodes with the CURRENT policy.
    2. Compute reward-to-go G_t for every step.
    3. Estimate the advantage: A_t = G_t - V(s_t), or A_t = G_t with no baseline.
    4. Take one gradient ASCENT step on  sum_t log pi(a_t|s_t) * A_t.
    5. Refit V on this batch's returns (after step 3 — see train/baseline.py).
    6. Optionally reuse the PREVIOUS batch with importance sampling.
    7. Write one `episodes` row per episode and one `gradient_stats` row per
       update.

What "gradient variance" means here
-----------------------------------
The batch estimate is the mean of `batch_episodes` independent per-episode
gradient estimates. We therefore compute those per-episode gradients
individually — one backward pass each — stack them into a matrix G of shape
(n_episodes, n_parameters), and report

    gradient_variance = sum_j Var_i(G[i, j])        (trace of the covariance)
    gradient_norm     = || mean_i G[i, :] ||        (the step actually taken)

The step taken is exactly `mean_i G[i, :]`, so the statistic describes the same
quantity the optimiser consumed. That is the point: it is tempting to log
`||grad||` after a single fused backward pass and call it variance, but a norm
is not a variance and one sample has none. The extra backward passes are the
price of being able to answer the client's question.

Scaling note: the trace of a covariance has units of (gradient)^2 and its
absolute value is not interpretable on its own. Compare arms, and compare them
at the same update index — which is what `gradient_variance_by_arm` in
db/migrations/002_gradient_stats.sql exists to make easy.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from envs import ENV_ID, N_ACTIONS, OBS_DIM, make_env
from train.baseline import (
    ValueNetwork,
    advantages,
    discounted_returns,
    explained_variance,
    fit_value_network,
)
from train.policy import DEFAULT_HIDDEN, CategoricalPolicy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class VPGConfig:
    """Every knob, in one place, and every one of them lands in
    `experiments.hyperparameters` as jsonb.

    That is not bookkeeping. Six weeks from now the only way to answer "what
    was different about run 7" is to read the row, and a hyperparameter that
    lived in a command-line default rather than in the database is a
    hyperparameter you will not recover.
    """

    env_id: str = ENV_ID
    episodes: int = 400
    seed: int = 0

    # A batch of complete EPISODES rather than a fixed number of steps. Monte
    # Carlo returns need the episode to have ended, and mixing a truncated
    # episode's partial return into the batch biases every advantage in it.
    batch_episodes: int = 10
    max_steps: int = 500  # CartPole-v1's own time limit; stated, not assumed

    gamma: float = 0.99
    # 1e-2 with Adam, found by running the three-value sweep Topic 3 DQ 4 asks
    # for and keeping the middle one. Note how few gradient steps a policy
    # gradient actually takes: 400 episodes at 10 per batch is FORTY updates,
    # so a learning rate tuned for supervised training (1e-4, say) leaves the
    # policy essentially where it started and looks like an implementation bug.
    policy_lr: float = 1e-2
    value_lr: float = 1e-2
    value_epochs: int = 40
    hidden: tuple[int, ...] = DEFAULT_HIDDEN

    # Off by default, and that is a measurement decision rather than a taste
    # one: an entropy bonus adds its own term to the gradient, and the headline
    # chart is supposed to show the variance of the POLICY GRADIENT, not of the
    # policy gradient plus a regulariser. Turn it on to explore Topic 3 DQ 6,
    # and say so when you report the numbers.
    entropy_coef: float = 0.0

    # -- the 2x2 --
    use_baseline: bool = True
    use_importance_sampling: bool = False

    # How many extra updates to take on the PREVIOUS batch when importance
    # sampling is on. One is enough to see the effect; more makes the ratio
    # drift further from 1 and the weights heavier, which is itself the lesson.
    is_reuse_updates: int = 1
    # Truncation ceiling for the importance ratio. Unbounded ratios are how
    # off-policy policy gradients blow up: one state where the old policy
    # assigned probability 0.01 and the new one assigns 0.9 contributes a weight
    # of 90 and swamps the batch. See the comment in `off_policy_update`.
    is_clip: float = 10.0

    def as_hyperparameters(self) -> dict[str, Any]:
        d = asdict(self)
        d["hidden"] = list(self.hidden)  # jsonb has no tuple type
        return d

    @property
    def arm(self) -> str:
        """The human name of this cell of the 2x2, used as `experiments.algorithm`."""
        return "vpg" + ("+baseline" if self.use_baseline else "") + (
            "+is" if self.use_importance_sampling else ""
        )


@dataclass
class Trajectory:
    """One complete episode, kept in the shape the update needs it.

    `behaviour_log_probs` is the field that earns its keep. It is log pi(a|s)
    under the policy that ACTUALLY COLLECTED this data, captured at collection
    time. Recomputing it later, after the policy has moved, would give an
    importance ratio of exactly 1 everywhere — an importance-sampling arm that
    runs, logs, plots, and does nothing. That bug is invisible in the return
    curve and obvious in the weight histogram, which is one reason the
    assignment asks you to plot the histogram.
    """

    obs: np.ndarray
    actions: np.ndarray
    rewards: list[float]
    behaviour_log_probs: np.ndarray
    returns: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def total_reward(self) -> float:
        return float(sum(self.rewards))

    @property
    def length(self) -> int:
        return len(self.rewards)


# ---------------------------------------------------------------------------
# Flat gradient plumbing
# ---------------------------------------------------------------------------


def flat_gradient(module: torch.nn.Module) -> np.ndarray:
    """Every parameter's gradient, concatenated into one vector.

    Flattened because variance is a statement about the estimator as a whole,
    and reporting a separate variance per weight tensor would give you four
    numbers to compare instead of one. `module.parameters()` yields in a stable
    order, so the same index means the same weight across calls — which is what
    makes stacking the per-episode vectors legitimate.
    """
    parts = [
        (p.grad.detach().cpu().numpy().ravel() if p.grad is not None else np.zeros(p.numel()))
        for p in module.parameters()
    ]
    return np.concatenate(parts) if parts else np.zeros(0)


def set_flat_gradient(module: torch.nn.Module, vec: np.ndarray) -> None:
    """Write a flat vector back into `.grad`, so `optimiser.step()` uses it.

    This is what lets us take the per-episode gradients we measured and descend
    their MEAN, rather than measuring one thing and stepping on another. If the
    logged variance describes a gradient the optimiser never saw, the chart is
    decoration.
    """
    i = 0
    for p in module.parameters():
        n = p.numel()
        g = torch.as_tensor(vec[i : i + n], dtype=p.dtype).view_as(p)
        p.grad = g.clone()
        i += n


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def collect_batch(
    env,
    policy: CategoricalPolicy,
    n_episodes: int,
    max_steps: int,
    gamma: float,
    seed_offset: int,
) -> list[Trajectory]:
    """Roll out `n_episodes` complete episodes under the current policy."""
    batch: list[Trajectory] = []
    for k in range(n_episodes):
        # A distinct seed per episode, derived from a counter rather than left
        # to chance. `env.reset()` with no seed uses whatever state the env
        # happens to hold, and a "seeded" run that reseeds nothing is not
        # reproducible however loudly the README says seed=0.
        obs, _ = env.reset(seed=seed_offset + k)
        obs_list, act_list, rew_list, logp_list = [], [], [], []
        for _ in range(max_steps):
            action, logp = policy.act(np.asarray(obs, dtype=np.float32))
            obs_list.append(np.asarray(obs, dtype=np.float32))
            act_list.append(action)
            logp_list.append(logp)
            obs, reward, terminated, truncated, _ = env.step(action)
            rew_list.append(float(reward))
            if terminated or truncated:
                break
        traj = Trajectory(
            obs=np.asarray(obs_list, dtype=np.float32),
            actions=np.asarray(act_list, dtype=np.int64),
            rewards=rew_list,
            behaviour_log_probs=np.asarray(logp_list, dtype=np.float64),
        )
        traj.returns = discounted_returns(rew_list, gamma)
        batch.append(traj)
    return batch


# ---------------------------------------------------------------------------
# The updates
# ---------------------------------------------------------------------------


def _batch_advantages(
    batch: list[Trajectory], value_net: ValueNetwork | None
) -> tuple[list[np.ndarray], float]:
    """Per-trajectory advantages, plus how much of the return V explained."""
    advs, all_returns, all_values = [], [], []
    for traj in batch:
        a, v = advantages(traj.returns, value_net, torch.as_tensor(traj.obs))
        advs.append(a)
        all_returns.append(traj.returns)
        all_values.append(v)
    ev = explained_variance(np.concatenate(all_returns), np.concatenate(all_values))
    return advs, ev


def on_policy_update(
    policy: CategoricalPolicy,
    optimiser: torch.optim.Optimizer,
    batch: list[Trajectory],
    advs: list[np.ndarray],
    entropy_coef: float,
) -> dict[str, Any]:
    """One ascent step on the vanilla policy gradient, with per-episode statistics.

    The objective maximised is

        J_i = sum_t log pi(a_t|s_t) * A_t  +  entropy_coef * sum_t H(pi(.|s_t))

    computed once per EPISODE i so that the spread across i can be measured.
    `loss = -J_i` because optimisers descend; ascending a gradient by descending
    its negation is the whole of "gradient ascent" in PyTorch, and writing it
    out is cheaper than the sign error you get from remembering it.
    """
    grads = []
    entropies = []
    for traj, adv in zip(batch, advs):
        optimiser.zero_grad(set_to_none=False)
        obs = torch.as_tensor(traj.obs)
        acts = torch.as_tensor(traj.actions)
        logp = policy.log_prob(obs, acts)
        ent = policy.entropy(obs)
        objective = (logp * torch.as_tensor(adv, dtype=torch.float32)).sum()
        objective = objective + entropy_coef * ent.sum()
        (-objective).backward()
        grads.append(flat_gradient(policy))
        entropies.append(float(ent.mean().item()))

    G = np.stack(grads)
    mean_g = G.mean(axis=0)
    set_flat_gradient(policy, mean_g)
    optimiser.step()

    return {
        "gradient_norm": float(np.linalg.norm(mean_g)),
        # ddof=1: these are `n` samples used to estimate a population variance,
        # and with n=10 the difference between dividing by 10 and by 9 is 11%.
        # Two arms compared with different ddof is a fake finding.
        "gradient_variance": float(G.var(axis=0, ddof=1).sum()) if len(G) > 1 else 0.0,
        "policy_entropy": float(np.mean(entropies)),
        "off_policy": False,
    }


def off_policy_update(
    policy: CategoricalPolicy,
    optimiser: torch.optim.Optimizer,
    batch: list[Trajectory],
    advs: list[np.ndarray],
    entropy_coef: float,
    is_clip: float,
    histogram_bins: int = 20,
) -> dict[str, Any]:
    """One ascent step that REUSES a batch collected under an older policy.

    The correction
    --------------
    Data drawn from pi_old can estimate an expectation under pi_new by
    reweighting:

        E_new[f] = E_old[ (pi_new/pi_old) * f ]

    which is unbiased whenever pi_old(a|s) > 0 wherever pi_new(a|s) > 0. The
    surrogate maximised here is

        J = sum_t  ratio_t * A_t,     ratio_t = pi_new(a_t|s_t) / pi_old(a_t|s_t)

    and its gradient is `ratio_t * grad log pi_new(a_t|s_t) * A_t` — the
    off-policy policy gradient, written in the form that lets autograd produce
    it. Note that A_t is recomputed from the CURRENT value network: the returns
    in the old batch are fixed facts, but the baseline is not.

    Per-step ratios, not per-trajectory
    -----------------------------------
    The textbook trajectory-level weight is the product of the per-step ratios
    over the whole episode. On a 200-step CartPole episode that product is a
    product of 200 numbers near 1, which is numerically either 0 or enormous —
    its variance grows exponentially in the horizon, and this is the standard
    reason ordinary importance sampling is unusable for long episodes. The
    per-step ratio used here is the approximation every practical method makes
    (PPO included). It is BIASED — say so in your README — and the bias buys a
    variance that does not explode with the horizon.

    Truncation
    ----------
    `clamp(ratio, max=is_clip)` bounds the weight. Note that clamping also kills
    the gradient for any sample above the ceiling: beyond `is_clip` the
    surrogate is flat, so the sample contributes nothing. That is not a side
    effect to work around, it IS the mechanism — and it is precisely what PPO's
    clipped objective does, two-sided and with a much tighter ceiling. If your
    weight histogram piles up against the ceiling, the two policies have drifted
    too far for this batch to be worth reusing, and the honest fix is to reuse
    fewer batches rather than to raise the ceiling.
    """
    grads, entropies, weights = [], [], []
    for traj, adv in zip(batch, advs):
        optimiser.zero_grad(set_to_none=False)
        obs = torch.as_tensor(traj.obs)
        acts = torch.as_tensor(traj.actions)
        logp_new = policy.log_prob(obs, acts)
        logp_old = torch.as_tensor(traj.behaviour_log_probs, dtype=torch.float32)

        # Exponentiate a CLAMPED log-ratio. Doing it the other way round —
        # exp() first, clamp after — overflows to inf for a log-ratio of 90
        # before the clamp ever runs, and inf * 0 is NaN.
        log_ratio = torch.clamp(logp_new - logp_old, max=math.log(is_clip))
        ratio = torch.exp(log_ratio)

        ent = policy.entropy(obs)
        objective = (ratio * torch.as_tensor(adv, dtype=torch.float32)).sum()
        objective = objective + entropy_coef * ent.sum()
        (-objective).backward()

        grads.append(flat_gradient(policy))
        entropies.append(float(ent.mean().item()))
        weights.append(ratio.detach().cpu().numpy())

    G = np.stack(grads)
    mean_g = G.mean(axis=0)
    set_flat_gradient(policy, mean_g)
    optimiser.step()

    w = np.concatenate(weights).astype(np.float64)
    # Effective sample size as a FRACTION of n. Reported this way because the
    # raw ESS is not comparable across batches of different length, and the
    # fraction is directly readable: 0.05 means twenty weighted samples are
    # doing the work of one.
    ess = float((w.sum() ** 2) / (len(w) * np.sum(w**2))) if np.sum(w**2) > 0 else 0.0
    counts, edges = np.histogram(w, bins=histogram_bins, range=(0.0, is_clip))

    return {
        "gradient_norm": float(np.linalg.norm(mean_g)),
        "gradient_variance": float(G.var(axis=0, ddof=1).sum()) if len(G) > 1 else 0.0,
        "policy_entropy": float(np.mean(entropies)),
        "off_policy": True,
        "is_weight_mean": float(w.mean()),
        "is_weight_max": float(w.max()),
        "is_weight_p95": float(np.percentile(w, 95)),
        "is_weight_ess": ess,
        # A histogram, not the raw weights. One run at the real budget produces
        # a few hundred thousand of them and the free tier has 500 MB; the
        # distribution is what the assignment asks you to plot, and twenty bins
        # draw the same picture.
        "is_weight_histogram": {
            "edges": [float(x) for x in edges],
            "counts": [int(c) for c in counts],
        },
    }


# ---------------------------------------------------------------------------
# The training run
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    experiment_id: str
    config: VPGConfig
    episode_returns: list[float]
    gradient_stats: list[dict[str, Any]]
    policy: CategoricalPolicy
    value_net: ValueNetwork | None
    explained_variance: float

    def mean_return_last(self, n: int = 50) -> float:
        """The headline learning number. Quote the window with it, always.

        "Mean return 180" is not a claim; "mean return over the last 50 of 400
        episodes, seed 0" is. A method rather than a stored field because the
        window is an argument to the claim, not a property of the run.
        """
        tail = self.episode_returns[-n:]
        return float(np.mean(tail)) if tail else 0.0


def train_vpg(
    cfg: VPGConfig,
    store: Any | None = None,
    log_every: int = 0,
) -> RunResult:
    """Train one arm, one seed, and write every row it is supposed to write.

    Telemetry goes to the store as the run proceeds, not at the end. A run that
    crashes at episode 900 of 1,000 should leave 900 usable rows behind, and a
    training script that buffers everything in memory and writes once leaves
    nothing.
    """
    from shared.store import get_store

    store = store or get_store()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    policy = CategoricalPolicy(OBS_DIM, N_ACTIONS, cfg.hidden, seed=cfg.seed)
    policy_opt = torch.optim.Adam(policy.parameters(), lr=cfg.policy_lr)

    value_net = value_opt = None
    if cfg.use_baseline:
        # Offset the value network's seed so its initial weights are independent
        # of the policy's — see the note in train/baseline.py.
        value_net = ValueNetwork(OBS_DIM, cfg.hidden, seed=cfg.seed + 10_000)
        value_opt = torch.optim.Adam(value_net.parameters(), lr=cfg.value_lr)

    experiment_id = store.insert_experiment(
        {
            "algorithm": cfg.arm,
            "env_id": cfg.env_id,
            "seed": cfg.seed,
            "hyperparameters": cfg.as_hyperparameters(),
        }
    )

    env = make_env()
    episode_returns: list[float] = []
    stat_rows: list[dict[str, Any]] = []
    previous_batch: list[Trajectory] | None = None
    update_index = 0
    ev = 0.0

    while len(episode_returns) < cfg.episodes:
        remaining = cfg.episodes - len(episode_returns)
        n = min(cfg.batch_episodes, remaining)

        batch = collect_batch(
            env,
            policy,
            n,
            cfg.max_steps,
            cfg.gamma,
            # Seeds advance with the episode counter so no two batches in a run
            # replay the same initial states, and so seed 0 and seed 1 never
            # overlap.
            seed_offset=cfg.seed * 1_000_000 + len(episode_returns),
        )

        first_index = len(episode_returns)
        episode_rows = []
        for i, traj in enumerate(batch):
            episode_returns.append(traj.total_reward)
            episode_rows.append(
                {
                    "experiment_id": experiment_id,
                    # Absolute index within the run, not within the batch. The
                    # unique constraint on (experiment_id, episode_index) in
                    # 001_init.sql will catch a per-batch counter immediately,
                    # which is exactly why that constraint is there.
                    "episode_index": first_index + i,
                    "return": traj.total_reward,
                    "length": traj.length,
                    # NULL, and honestly so. A softmax policy has no epsilon;
                    # exploration lives in the entropy of the distribution,
                    # which is logged per update in gradient_stats instead.
                    "epsilon": None,
                }
            )
        store.insert_episodes(episode_rows)

        advs, ev = _batch_advantages(batch, value_net)
        stats = on_policy_update(policy, policy_opt, batch, advs, cfg.entropy_coef)
        stat_rows.append(
            {
                "experiment_id": experiment_id,
                "update_index": update_index,
                "episode_index": len(episode_returns),
                **stats,
            }
        )
        update_index += 1

        # Value network AFTER the advantage, never before. See the ordering
        # argument in train/baseline.fit_value_network.
        if value_net is not None and value_opt is not None:
            all_obs = torch.as_tensor(np.concatenate([t.obs for t in batch]))
            all_ret = np.concatenate([t.returns for t in batch])
            fit_value_network(value_net, value_opt, all_obs, all_ret, cfg.value_epochs)

        if cfg.use_importance_sampling and previous_batch is not None:
            for _ in range(cfg.is_reuse_updates):
                # Advantages recomputed against the CURRENT value network: the
                # old returns are fixed, the baseline has moved on.
                old_advs, _ = _batch_advantages(previous_batch, value_net)
                stats = off_policy_update(
                    policy, policy_opt, previous_batch, old_advs, cfg.entropy_coef, cfg.is_clip
                )
                stat_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "update_index": update_index,
                        # Same episode_index as the on-policy update above: this
                        # update consumed NO new environment interaction, which
                        # is the entire point of off-policy reuse and must be
                        # visible in the x-axis of the sample-efficiency plot.
                        "episode_index": len(episode_returns),
                        **stats,
                    }
                )
                update_index += 1

        previous_batch = batch

        if log_every and len(episode_returns) % log_every < n:
            tail = episode_returns[-50:]
            print(
                f"[{cfg.arm} seed={cfg.seed}] ep {len(episode_returns):5d}  "
                f"mean50={np.mean(tail):7.1f}  "
                f"gvar={stat_rows[-1]['gradient_variance']:.4g}  "
                f"H={stat_rows[-1]['policy_entropy']:.3f}  ev={ev:+.2f}",
                flush=True,
            )

    env.close()
    store.insert_gradient_stats(stat_rows)

    return RunResult(
        experiment_id=experiment_id,
        config=cfg,
        episode_returns=episode_returns,
        gradient_stats=stat_rows,
        policy=policy,
        value_net=value_net,
        explained_variance=ev,
    )


def evaluate(policy: CategoricalPolicy, episodes: int = 20, seed: int = 12345) -> dict[str, float]:
    """Greedy evaluation. A different question from the training curve.

    The training curve is what the STOCHASTIC policy scored while it was still
    exploring; this is what the deployed policy scores. Reporting the first as
    though it were the second is the most common overstatement in a first RL
    report, which is why `evaluations` is its own table.
    """
    env = make_env()
    returns = []
    for k in range(episodes):
        obs, _ = env.reset(seed=seed + k)
        total = 0.0
        for _ in range(500):
            with torch.no_grad():
                logits = policy(torch.as_tensor(np.asarray(obs, dtype=np.float32)))
            obs, reward, terminated, truncated, _ = env.step(int(torch.argmax(logits).item()))
            total += float(reward)
            if terminated or truncated:
                break
        returns.append(total)
    env.close()
    arr = np.asarray(returns, dtype=np.float64)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return {
        "episodes": float(episodes),
        "mean_return": float(arr.mean()),
        "std_return": std,
        "stderr_return": std / math.sqrt(max(len(arr), 1)),
    }


# ---------------------------------------------------------------------------
# The controlled experiment behind the baseline claim
# ---------------------------------------------------------------------------


def compare_baseline_variance(
    seed: int = 0,
    batch_episodes: int = 12,
    gamma: float = 0.99,
    value_epochs: int = 200,
) -> dict[str, float]:
    """Measure gradient variance with and without a baseline on ONE batch.

    The controls matter more than the measurement here:

      * The SAME trajectories are used for both arms. Collect two batches and
        you are measuring the seed as much as the baseline, and with a batch of
        twelve episodes the seed wins often enough to produce a confident wrong
        answer.
      * The SAME policy parameters produce both gradients. The advantage is the
        only thing that changes between the two numbers.
      * The value network is fitted on this batch's returns before use, which is
        optimistic — inside the real loop V is fitted on the PREVIOUS batch (see
        train/vpg.train_vpg). The optimism is deliberate here: this function
        answers "does subtracting a good baseline reduce variance", which is the
        theoretical claim, not "how well does my value network generalise",
        which is a separate empirical question with its own column
        (`explained_variance`).

    Used by tests/test_baseline_reduces_variance.py and by the README's
    quantitative section.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = CategoricalPolicy(OBS_DIM, N_ACTIONS, seed=seed)
    env = make_env()
    batch = collect_batch(env, policy, batch_episodes, 500, gamma, seed_offset=seed * 1000)
    env.close()

    value_net = ValueNetwork(OBS_DIM, seed=seed + 10_000)
    value_opt = torch.optim.Adam(value_net.parameters(), lr=1e-2)
    all_obs = torch.as_tensor(np.concatenate([t.obs for t in batch]))
    all_ret = np.concatenate([t.returns for t in batch])
    fit_value_network(value_net, value_opt, all_obs, all_ret, value_epochs)

    out: dict[str, float] = {}
    for label, vnet in (("without_baseline", None), ("with_baseline", value_net)):
        advs, ev = _batch_advantages(batch, vnet)
        grads = []
        for traj, adv in zip(batch, advs):
            policy.zero_grad(set_to_none=False)
            logp = policy.log_prob(torch.as_tensor(traj.obs), torch.as_tensor(traj.actions))
            (-(logp * torch.as_tensor(adv, dtype=torch.float32)).sum()).backward()
            grads.append(flat_gradient(policy))
        G = np.stack(grads)
        out[f"{label}_variance"] = float(G.var(axis=0, ddof=1).sum())
        out[f"{label}_norm"] = float(np.linalg.norm(G.mean(axis=0)))
        out[f"{label}_explained_variance"] = float(ev)

    # NOTE: the policy is never stepped in this function, so the two gradients
    # are taken at identical parameters. If you add an optimiser step here the
    # comparison stops being controlled.
    out["variance_ratio"] = (
        out["without_baseline_variance"] / out["with_baseline_variance"]
        if out["with_baseline_variance"] > 0
        else float("inf")
    )
    return out
