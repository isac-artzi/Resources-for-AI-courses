"""
train/sac.py — soft actor-critic on Pendulum-v1.

TRAINING TIER (imports torch). Never reachable from api/ or ui/.

SAC is where the topic's argument lands. A2C and PPO treat exploration as
something bolted onto the loss: an entropy bonus with a coefficient, which
decays in importance as the policy sharpens and which the optimal policy of the
underlying MDP does not contain. SAC changes the OBJECTIVE instead:

    J(π) = E[ Σ_t γ^t ( r(s_t, a_t) + α·H(π(·|s_t)) ) ]

Entropy is now part of the return. The optimal policy of THIS objective is
stochastic — it is not a deterministic policy plus noise — and that is the
difference the entropy sweep in `train/entropy_sweep.py` is designed to expose.

What follows from the objective
-------------------------------
Every piece of SAC is a consequence of that one change:

**Soft value functions.** With entropy inside the reward, the Bellman backup
picks up a −log π term:

    Q(s,a) = r + γ·E_{s'}[ V(s') ],    V(s) = E_{a'~π}[ Q(s',a') − α·log π(a'|s') ]

so the target computed in `_critic_loss` below is
`r + γ(1−done)(min(Q1,Q2)(s',a') − α·log π(a'|s'))`. The `− α log π` is the
entropy term of the soft Bellman equation and it is not optional: drop it and
you have TD3 with a stochastic actor.

**The soft policy gradient.** The actor maximises
`E_{a~π}[ Q(s,a) − α·log π(a|s) ]`, differentiated through the reparameterised
sample. That is why `SquashedGaussianActor.sample` uses `rsample()` — the
gradient has to flow through the action into the critic.

**Automatic α.** α is a Lagrange multiplier on a constraint "average entropy at
least H̄", and it can be solved for rather than guessed:

    L(α) = −α·( log π(a|s) + H̄ ),     H̄ = −dim(A)   by convention

This matters more than it looks. α trades off against a REWARD SCALE, so the
"right" α on Pendulum (rewards in [−16, 0]) is not the right α on a task whose
rewards are in [0, 1] — it would have to be rescaled by roughly the same factor.
A fixed α is therefore a hyperparameter you must retune for every environment,
and worse, the right value CHANGES DURING TRAINING: early on the agent should be
uncertain, later it should not. That is the argument for automatic tuning, and
the entropy sweep is where you check whether it holds on your own data.

Budget note
-----------
SAC takes one gradient update per environment step, so it is far more expensive
per step than A2C or PPO and far cheaper per unit of PROGRESS. Pendulum reaches
a usable policy in roughly 10,000 steps, which is fifty episodes. Do not compare
SAC's wall clock with PPO's and call it sample efficiency — they are different
axes, and this product's comparisons are on environment steps.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from envs import make_env, spec_for
from train.nets import CONTINUOUS_HIDDEN, SquashedGaussianActor, TwinQ


@dataclass
class SACConfig:
    """Every knob, in one place, and every one lands in `experiments.hyperparameters`."""

    env_id: str = "Pendulum-v1"
    total_steps: int = 12_000
    seed: int = 0

    gamma: float = 0.99
    # Polyak averaging coefficient for the target critics: θ_targ ← τ·θ + (1−τ)·θ_targ.
    # 0.005 means the target moves 0.5% of the way to the online critic per step,
    # i.e. a time constant of about 200 updates. Its job is to make the
    # bootstrap target slow-moving; a target that chases the online network
    # exactly is a network regressing on its own output, which diverges.
    tau: float = 0.005
    lr: float = 3e-4
    hidden: tuple[int, ...] = CONTINUOUS_HIDDEN

    batch_size: int = 256
    replay_size: int = 100_000
    # Uniformly RANDOM actions for the first `start_steps`. Not a heuristic
    # bolted on: at initialisation the actor's tanh-squashed Gaussian is
    # concentrated near zero torque, so a Pendulum that starts hanging down
    # simply hangs there, and the replay buffer fills with a few hundred copies
    # of one state. Random actions buy state coverage that the policy cannot
    # produce yet.
    start_steps: int = 1_000
    update_after: int = 1_000     # do not fit a critic on 200 correlated transitions
    update_every: int = 1         # one gradient update per environment step

    # α, the entropy temperature. Either a fixed number or tuned automatically
    # against a target entropy. `alpha` is the FIXED value when auto is off, and
    # the INITIAL value when it is on.
    alpha: float = 0.2
    auto_alpha: bool = True
    # H̄ = −dim(A) is the convention from the SAC paper. For Pendulum that is
    # −1.0. It is a convention, not a derivation — a target differential entropy
    # has units and a scale, and −dim(A) happens to work across the MuJoCo suite.
    # Set it explicitly if you want to argue for a different value; leaving it
    # None takes the convention and records that fact in the hyperparameters.
    target_entropy: float | None = None

    @property
    def algorithm(self) -> str:
        return "sac"

    @property
    def alpha_setting(self) -> str:
        """The human label of this arm, used by the entropy sweep and the UI."""
        return "auto" if self.auto_alpha else f"alpha={self.alpha:g}"

    def as_hyperparameters(self) -> dict[str, Any]:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        d["alpha_setting"] = self.alpha_setting
        return d


@dataclass
class ReplayBuffer:
    """A fixed-size ring of transitions, in NumPy.

    NumPy rather than a list of tuples, and preallocated rather than appended.
    A 100,000-step buffer of Python tuples is 100,000 objects the garbage
    collector walks, and sampling a minibatch from it means 256 random list
    indexes and 256 tuple unpacks per environment step. Preallocated arrays make
    a minibatch one fancy-index per field.

    This buffer is what makes SAC OFF-POLICY, and off-policy is the reason it
    needs so many fewer environment steps than PPO: every transition is reused
    for many gradient updates instead of being discarded after one iteration.
    The price is that the data is stale — collected under policies the agent no
    longer follows — which is exactly what the twin critics and the target
    networks exist to keep stable.
    """

    obs_dim: int
    act_dim: int
    capacity: int
    obs: np.ndarray = field(init=False)
    act: np.ndarray = field(init=False)
    rew: np.ndarray = field(init=False)
    next_obs: np.ndarray = field(init=False)
    done: np.ndarray = field(init=False)
    size: int = field(default=0, init=False)
    ptr: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.act = np.zeros((self.capacity, self.act_dim), dtype=np.float32)
        self.rew = np.zeros(self.capacity, dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.done = np.zeros(self.capacity, dtype=np.float32)

    def add(self, obs, act, rew, next_obs, done) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        # `done` here must be TERMINATED only. Pendulum never terminates — it
        # only truncates at 200 steps — so storing `terminated or truncated`
        # would mark every 200th transition as terminal and teach the critic
        # that the world ends there. On Pendulum that single mistake is the
        # difference between converging to −200 and plateauing around −900.
        self.done[i] = float(done)
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        idx = rng.integers(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx]),
            "act": torch.as_tensor(self.act[idx]),
            "rew": torch.as_tensor(self.rew[idx]),
            "next_obs": torch.as_tensor(self.next_obs[idx]),
            "done": torch.as_tensor(self.done[idx]),
        }


@dataclass
class SACRunResult:
    experiment_id: str
    config: SACConfig
    episode_returns: list[float]
    episode_steps: list[int]
    updates: list[dict[str, Any]]
    actor: SquashedGaussianActor
    final_alpha: float
    mean_policy_entropy: float

    def mean_return_last(self, n: int = 100) -> float:
        tail = self.episode_returns[-n:]
        return float(np.mean(tail)) if tail else 0.0

    def std_return_last(self, n: int = 100) -> float:
        tail = self.episode_returns[-n:]
        return float(np.std(tail, ddof=1)) if len(tail) > 1 else 0.0

    def episodes_to_threshold(self, threshold: float, window: int = 10) -> int | None:
        """First episode whose trailing `window` mean reached `threshold`, or None.

        None rather than a sentinel: a run that never got there is a result, and
        encoding it as 9999 puts a number into the mean of the column that is
        not a measurement of anything.
        """
        r = self.episode_returns
        for i in range(window - 1, len(r)):
            if float(np.mean(r[i - window + 1 : i + 1])) >= threshold:
                return i
        return None


def train_sac(
    cfg: SACConfig,
    store: Any | None = None,
    log_every: int = 10,
    update_log_every: int = 50,
) -> SACRunResult:
    """Train one SAC run: episode returns, critic loss and mean policy entropy.

    `update_log_every` exists because SAC takes one gradient update per
    environment step. A 12,000-step run would write 11,000 `policy_updates` rows
    if every update were stored, and the report budget is several times that
    across nine sweep runs — which is a meaningful fraction of a 500 MB free
    tier for a series no chart can resolve. Every 50th update is stored, and the
    stored row carries the MEAN of the intervening updates rather than a single
    sampled one, so the series is a summary and not a subsample. Say which in
    your write-up.
    """
    from shared.store import get_store

    store = store or get_store()
    spec = spec_for(cfg.env_id)
    if spec.discrete:
        raise ValueError(
            f"{cfg.env_id} has a discrete action space. This SAC implementation is the "
            "continuous one; discrete SAC replaces the squashed Gaussian with a "
            "categorical actor and computes the entropy exactly rather than from a sample."
        )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    act_dim = spec.n_actions
    actor = SquashedGaussianActor(
        spec.obs_dim,
        act_dim,
        action_scale=spec.action_high,
        action_bias=0.0,
        hidden=cfg.hidden,
        seed=cfg.seed,
    )
    critic = TwinQ(spec.obs_dim, act_dim, cfg.hidden, seed=cfg.seed + 10_000)
    # A frozen COPY, not a second construction. The target must start identical
    # to the online critic; two networks built from different seeds would make
    # the first thousand bootstrap targets pure noise, and SAC would spend the
    # early part of every run recovering from it.
    critic_target = copy.deepcopy(critic)
    for p in critic_target.parameters():
        p.requires_grad_(False)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.lr)

    target_entropy = cfg.target_entropy if cfg.target_entropy is not None else -float(act_dim)
    # log α rather than α, optimised without a constraint. α must be positive;
    # optimising it directly would need a projection every step, and one
    # unlucky gradient would make it negative — at which point the objective
    # rewards LOW entropy and the policy collapses to a deterministic one
    # within a few hundred updates, with no error anywhere.
    log_alpha = torch.tensor(float(np.log(cfg.alpha)), requires_grad=cfg.auto_alpha)
    alpha_opt = torch.optim.Adam([log_alpha], lr=cfg.lr) if cfg.auto_alpha else None

    experiment_id = store.insert_experiment(
        {
            "algorithm": cfg.algorithm,
            "env_id": cfg.env_id,
            "seed": cfg.seed,
            "hyperparameters": cfg.as_hyperparameters(),
        }
    )

    buffer = ReplayBuffer(spec.obs_dim, act_dim, cfg.replay_size)
    env = make_env(cfg.env_id)
    seed_base = cfg.seed * 1_000_000
    episode_index = 0
    obs, _ = env.reset(seed=seed_base)

    episode_returns: list[float] = []
    episode_steps: list[int] = []
    update_rows: list[dict[str, Any]] = []
    pending: list[dict[str, float]] = []
    all_entropies: list[float] = []
    episode_return, episode_length = 0.0, 0
    next_console_log = 0
    update_index = 0

    for step in range(1, cfg.total_steps + 1):
        if step <= cfg.start_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                a, _ = actor.sample(torch.as_tensor(np.asarray(obs, dtype=np.float32)))
            action = a.numpy()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        episode_return += float(reward)
        episode_length += 1
        # Only `terminated`. See the note in ReplayBuffer.add — this is the
        # single most consequential line in the file for Pendulum, which never
        # terminates and truncates every 200 steps.
        buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs

        if terminated or truncated:
            episode_returns.append(episode_return)
            episode_steps.append(step)
            store.insert_episodes(
                [
                    {
                        "experiment_id": experiment_id,
                        "episode_index": episode_index,
                        "return": episode_return,
                        "length": episode_length,
                        # Null: SAC has no epsilon. Its exploration IS the
                        # entropy of its own policy, which is what α prices.
                        "epsilon": None,
                        "env_steps": step,
                    }
                ]
            )
            episode_index += 1
            obs, _ = env.reset(seed=seed_base + episode_index)
            episode_return, episode_length = 0.0, 0

        if step >= cfg.update_after and step % cfg.update_every == 0:
            stats = _update(
                actor, critic, critic_target, actor_opt, critic_opt,
                log_alpha, alpha_opt, buffer, cfg, target_entropy, rng,
            )
            pending.append(stats)
            all_entropies.append(stats["policy_entropy"])
            if len(pending) >= update_log_every:
                update_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "update_index": update_index,
                        "env_steps": step,
                        "episode_index": episode_index,
                        "policy_loss": float(np.mean([p["policy_loss"] for p in pending])),
                        "value_loss": float(np.mean([p["value_loss"] for p in pending])),
                        "policy_entropy": float(np.mean([p["policy_entropy"] for p in pending])),
                        "kl_divergence": None,   # SAC has no trust region; see the module docstring
                        "clip_fraction": None,
                        "alpha": float(np.mean([p["alpha"] for p in pending])),
                    }
                )
                update_index += 1
                pending.clear()

        if log_every and episode_returns and len(episode_returns) >= next_console_log:
            next_console_log = len(episode_returns) + log_every
            tail = episode_returns[-10:]
            last = update_rows[-1] if update_rows else {}
            print(
                f"[sac {cfg.env_id} {cfg.alpha_setting} seed={cfg.seed}] "
                f"steps {step:6d}  ep {len(episode_returns):4d}  "
                f"mean10={np.mean(tail):8.1f}  "
                f"critic={last.get('value_loss', float('nan')):7.2f}  "
                f"H={last.get('policy_entropy', float('nan')):+.3f}  "
                f"alpha={last.get('alpha', float('nan')):.4f}",
                flush=True,
            )

    env.close()
    if pending:
        update_rows.append(
            {
                "experiment_id": experiment_id,
                "update_index": update_index,
                "env_steps": cfg.total_steps,
                "episode_index": episode_index,
                "policy_loss": float(np.mean([p["policy_loss"] for p in pending])),
                "value_loss": float(np.mean([p["value_loss"] for p in pending])),
                "policy_entropy": float(np.mean([p["policy_entropy"] for p in pending])),
                "kl_divergence": None,
                "clip_fraction": None,
                "alpha": float(np.mean([p["alpha"] for p in pending])),
            }
        )
    store.insert_policy_updates(update_rows)

    return SACRunResult(
        experiment_id=experiment_id,
        config=cfg,
        episode_returns=episode_returns,
        episode_steps=episode_steps,
        updates=update_rows,
        actor=actor,
        final_alpha=float(log_alpha.exp().item()),
        mean_policy_entropy=float(np.mean(all_entropies)) if all_entropies else float("nan"),
    )


def _update(
    actor, critic, critic_target, actor_opt, critic_opt,
    log_alpha, alpha_opt, buffer, cfg, target_entropy, rng,
) -> dict[str, float]:
    """One SAC update: critics, then actor, then α, then the polyak step.

    The ORDER is not arbitrary. The actor loss reads the critic, so updating the
    critic first means the actor is improved against the freshest value
    estimate. The α loss reads the actor's log-probability, so it comes after.
    And the target network moves last, once, per update — moving it inside the
    critic step would make the target chase within a single update.
    """
    batch = buffer.sample(cfg.batch_size, rng)
    alpha = log_alpha.exp().detach()

    # -- critics -----------------------------------------------------------
    with torch.no_grad():
        next_action, next_logp = actor.sample(batch["next_obs"])
        q1_t, q2_t = critic_target(batch["next_obs"], next_action)
        # The SOFT backup. `min(Q1, Q2)` is the twin-critic underestimate; the
        # `− α·log π` term is what makes this the soft Bellman equation rather
        # than the ordinary one. Delete it and you have TD3 with a stochastic
        # actor: the entropy would still appear in the actor's loss, but the
        # VALUE function would no longer account for it, and the two halves of
        # the algorithm would be optimising different objectives.
        target_v = torch.min(q1_t, q2_t) - alpha * next_logp
        target_q = batch["rew"] + cfg.gamma * (1.0 - batch["done"]) * target_v

    q1, q2 = critic(batch["obs"], batch["act"])
    critic_loss = torch.nn.functional.mse_loss(q1, target_q) + torch.nn.functional.mse_loss(
        q2, target_q
    )
    critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_opt.step()

    # -- actor -------------------------------------------------------------
    # The critic's parameters are frozen for the actor's backward pass. Without
    # this the actor's loss would also send gradients into the critic — pushing
    # Q towards being easy to maximise rather than towards being correct — and
    # the whole thing degenerates. `critic_opt.zero_grad()` above would not
    # save you, because the actor step happens before the next critic step.
    for p in critic.parameters():
        p.requires_grad_(False)

    new_action, logp = actor.sample(batch["obs"])
    q1_pi, q2_pi = critic(batch["obs"], new_action)
    # The soft policy gradient objective: maximise Q − α·log π, i.e. minimise
    # α·log π − Q. The second term is the entropy: −E[log π] IS the entropy of
    # the policy, estimated from the same reparameterised sample the Q term uses.
    actor_loss = (alpha * logp - torch.min(q1_pi, q2_pi)).mean()
    actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_opt.step()

    for p in critic.parameters():
        p.requires_grad_(True)

    # -- temperature -------------------------------------------------------
    if alpha_opt is not None:
        # L(α) = −α·(log π + H̄), evaluated at the CURRENT policy's log π,
        # detached. If the policy is more random than the target (log π below
        # −H̄) the gradient pushes α down; if it has become too deterministic,
        # α rises and buys entropy back. That feedback is the whole argument for
        # automatic tuning: a fixed α cannot respond to a policy that has
        # sharpened.
        alpha_loss = -(log_alpha.exp() * (logp.detach() + target_entropy)).mean()
        alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_opt.step()

    # -- target networks ---------------------------------------------------
    with torch.no_grad():
        for p, p_targ in zip(critic.parameters(), critic_target.parameters()):
            # In-place, under no_grad. `p_targ = τp + (1−τ)p_targ` written as an
            # assignment would rebind the loop variable and leave the target
            # network untouched — a bug that produces a slowly diverging critic
            # and no error message.
            p_targ.mul_(1.0 - cfg.tau).add_(cfg.tau * p)

    return {
        "policy_loss": float(actor_loss.item()),
        "value_loss": float(critic_loss.item()),
        # −E[log π] is the differential entropy of the squashed Gaussian,
        # estimated from this batch's samples. NEGATIVE values are normal and
        # correct: a continuous density can exceed 1, so differential entropy is
        # not bounded below by zero the way a discrete entropy is. Do not put
        # this on the same axis as a categorical policy's entropy without saying
        # so on the chart.
        "policy_entropy": float(-logp.mean().item()),
        "alpha": float(log_alpha.exp().item()),
    }


def evaluate(actor, env_id: str, episodes: int = 10, seed: int = 12345) -> dict[str, Any]:
    """Deterministic (modal) evaluation. A different question from the curve.

    The training curve is what the STOCHASTIC policy scored while it was still
    exploring — and under SAC it explores by construction, not by accident, so
    the gap between this number and the curve is larger than it would be for
    PPO and larger still at α = 0.5. That gap is itself a finding worth
    reporting from the sweep.
    """
    spec = spec_for(env_id)
    env = make_env(env_id)
    returns = []
    for k in range(episodes):
        obs, _ = env.reset(seed=seed + k)
        total = 0.0
        for _ in range(spec.max_steps):
            with torch.no_grad():
                a = actor.deterministic_action(torch.as_tensor(np.asarray(obs, dtype=np.float32)))
            obs, reward, terminated, truncated, _ = env.step(a.numpy())
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
