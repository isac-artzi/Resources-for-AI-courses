"""
train/ppo.py — proximal policy optimisation, with a clipped surrogate, on Acrobot-v1.

TRAINING TIER (imports torch). Never reachable from api/ or ui/.

PPO is A2C plus one idea: **reuse the batch, but only as far as you can trust
it.** A2C takes one gradient step per collected batch and throws it away,
because after one step the policy is no longer the policy that collected the
data and the estimator stops being valid. PPO reuses the same batch for several
epochs, correcting with an importance ratio, and stops the update from moving
the policy so far that the correction becomes meaningless.

The objective
-------------
    r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)

    L = E[ min( r_t·A_t ,  clip(r_t, 1−ε, 1+ε)·A_t ) ]

The `min` and the `clip` do something subtler than "bound the ratio", and this
is worth being able to explain:

  * When A_t > 0 (the action was better than average), the objective is capped
    at (1+ε)·A_t. Pushing the probability higher than that yields no further
    gain, so the gradient is ZERO beyond the ceiling. The update stops.
  * When A_t < 0, the `min` picks the LOWER of the two, which is the unclipped
    term once r_t drops below 1−ε. So an action that was much worse than average
    can still be pushed down without limit — the clip does not protect a bad
    action from being un-learned, only a good one from being over-learned.
  * The result is a one-sided, per-sample, first-order trust region enforced by
    making the objective FLAT rather than by constraining anything. Nothing here
    computes a KL, nothing here solves a constrained problem. That is precisely
    what distinguishes PPO from TRPO, and it is why the KL has to be MEASURED
    rather than assumed.

Which is what `kl_divergence` in `policy_updates` is for. TRPO enforces
KL(π_old ‖ π_new) ≤ δ as a hard constraint by solving a constrained optimisation
with a conjugate-gradient step and a line search. PPO clips a ratio and *hopes*
the KL stays small. Whether the hope held on YOUR run is an empirical question,
and this file logs the answer once per update so that Topic 4's DQ 3 can be
answered with a plot of your own data rather than with a paraphrase of the paper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from envs import spec_for
from train.a2c import RunResult
from train.nets import DISCRETE_HIDDEN, CategoricalActor, ValueCritic, generalised_advantage
from train.onpolicy import StepCollector, episode_rows, evaluate_discrete


@dataclass
class PPOConfig:
    """Every knob, in one place, and every one lands in `experiments.hyperparameters`."""

    env_id: str = "Acrobot-v1"
    total_steps: int = 60_000
    seed: int = 0

    # 1,024 steps per iteration, then ten epochs over it in minibatches of 256.
    # Longer batches than A2C's 128 on purpose: PPO's whole premise is that a
    # batch is worth reusing, so it wants a batch big enough for forty minibatch
    # updates to be a sensible thing to do to it. At 60,000 steps that is 58
    # iterations and 2,340 gradient steps, against A2C's 469 — and the two
    # consumed exactly the same amount of environment experience. That gap is
    # the sample-efficiency story of this topic in one sentence.
    n_steps: int = 1024
    epochs: int = 10
    minibatch_size: int = 256

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2       # ε. The trust region, such as it is.
    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    hidden: tuple[int, ...] = DISCRETE_HIDDEN

    # Acrobot needs a real entropy bonus and CartPole does not. The reward is
    # −1 per step until the goal is reached, so until the agent stumbles into
    # the goal ONCE every action looks equally bad and the advantage carries
    # almost no signal. Without the bonus the policy sharpens on noise, stops
    # exploring, and sits at −500 forever. This is the clearest example in the
    # course of exploration being a property of the objective rather than of a
    # separate mechanism.
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalise_advantages: bool = True

    # Optional early exit from the epoch loop when the measured KL exceeds this.
    # None by default: with it off, the KL column records what the clip ALONE
    # achieved, which is the honest measurement for the DQ 3 comparison against
    # TRPO. Set it to, say, 0.015 to add a genuine (if crude) trust region on
    # top of the clip, and then say in your write-up which of the two produced
    # the curve you are showing.
    target_kl: float | None = None

    @property
    def algorithm(self) -> str:
        return "ppo"

    def as_hyperparameters(self) -> dict[str, Any]:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        return d


def train_ppo(cfg: PPOConfig, store: Any | None = None, log_every: int = 20) -> RunResult:
    """Train one PPO run: one `episodes` row per episode, one `policy_updates` row per iteration."""
    from shared.store import get_store

    store = store or get_store()
    spec = spec_for(cfg.env_id)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    actor = CategoricalActor(spec.obs_dim, spec.n_actions, cfg.hidden, seed=cfg.seed)
    critic = ValueCritic(spec.obs_dim, cfg.hidden, seed=cfg.seed + 10_000)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.policy_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.value_lr)

    experiment_id = store.insert_experiment(
        {
            "algorithm": cfg.algorithm,
            "env_id": cfg.env_id,
            "seed": cfg.seed,
            "hyperparameters": cfg.as_hyperparameters(),
        }
    )

    collector = StepCollector(cfg.env_id, cfg.seed)
    episode_returns: list[float] = []
    episode_steps: list[int] = []
    update_rows: list[dict[str, Any]] = []
    update_index = 0
    next_console_log = 0
    rng = np.random.default_rng(cfg.seed)

    while collector.env_steps < cfg.total_steps:
        n = min(cfg.n_steps, cfg.total_steps - collector.env_steps)
        batch = collector.collect(actor, critic, n)

        adv, targets = generalised_advantage(
            batch.rewards,
            batch.values,
            batch.dones,
            batch.truncateds,
            batch.last_value,
            cfg.gamma,
            cfg.gae_lambda,
        )
        if cfg.normalise_advantages and len(adv) > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.as_tensor(batch.obs)
        act_t = torch.as_tensor(batch.actions)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        tgt_t = torch.as_tensor(targets, dtype=torch.float32)
        # log π_old, captured AT COLLECTION TIME by CategoricalActor.act(). This
        # is the field that earns its keep in this whole file: recomputing it
        # after the first epoch would give a ratio of exactly 1 everywhere — a
        # clipped surrogate that runs, logs, plots, and does nothing. The bug is
        # invisible in the return curve and obvious in the clip fraction, which
        # is one reason clip_fraction is a logged column.
        old_logp_t = torch.as_tensor(batch.log_probs, dtype=torch.float32)

        clip_fractions: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        stopped_early = False

        idx = np.arange(len(adv))
        for _epoch in range(cfg.epochs):
            rng.shuffle(idx)
            for start in range(0, len(idx), cfg.minibatch_size):
                mb = idx[start : start + cfg.minibatch_size]
                if len(mb) < 2:
                    continue  # a minibatch of one has no advantage spread to learn from
                mb_t = torch.as_tensor(mb)

                logp = actor.log_prob(obs_t[mb_t], act_t[mb_t])
                # exp of a DIFFERENCE of log-probabilities, never a quotient of
                # probabilities. π_old(a|s) can be 1e-8 for an action the policy
                # has learned to avoid, and dividing by it overflows; the log
                # difference is a small finite number and exp() of it is stable.
                ratio = torch.exp(logp - old_logp_t[mb_t])
                mb_adv = adv_t[mb_t]

                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * mb_adv
                # `min` of the two, then negate: we ascend the PESSIMISTIC bound
                # on the improvement. Taking the max instead — a sign slip that
                # still runs — turns PPO into a method that actively seeks the
                # largest possible policy change, and it diverges spectacularly.
                policy_loss = -torch.min(unclipped, clipped).mean()

                entropy = actor.entropy(obs_t[mb_t]).mean()
                loss = policy_loss - cfg.entropy_coef * entropy

                value_loss = torch.nn.functional.mse_loss(critic(obs_t[mb_t]), tgt_t[mb_t])

                actor_opt.zero_grad(set_to_none=True)
                critic_opt.zero_grad(set_to_none=True)
                (loss + cfg.value_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
                actor_opt.step()
                critic_opt.step()

                with torch.no_grad():
                    clip_fractions.append(
                        float(((ratio - 1.0).abs() > cfg.clip_ratio).float().mean().item())
                    )
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))

            if cfg.target_kl is not None:
                if _mean_kl(actor, obs_t, act_t, old_logp_t) > cfg.target_kl:
                    stopped_early = True
                    break

        # THE measurement. Taken once, over the WHOLE batch, after all the
        # epochs — because the question PPO's clip is supposed to answer is "how
        # far did this iteration move the policy away from the one that
        # collected the data", and that is a property of the iteration, not of
        # any single minibatch. A KL logged per minibatch would give forty
        # numbers per iteration, none of which answers it.
        kl = _mean_kl(actor, obs_t, act_t, old_logp_t)

        first_index = len(episode_returns)
        rows = episode_rows(
            experiment_id, first_index, batch.finished_episodes, collector.env_steps
        )
        if rows:
            store.insert_episodes(rows)
            episode_returns.extend(r["return"] for r in rows)
            episode_steps.extend(r["env_steps"] for r in rows)

        update_rows.append(
            {
                "experiment_id": experiment_id,
                "update_index": update_index,
                "env_steps": collector.env_steps,
                "episode_index": len(episode_returns),
                "policy_loss": float(np.mean(policy_losses)) if policy_losses else None,
                "value_loss": float(np.mean(value_losses)) if value_losses else None,
                "policy_entropy": float(np.mean(entropies)) if entropies else None,
                "kl_divergence": kl,
                "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else None,
                "alpha": None,  # PPO has no entropy temperature; the bonus is a loss term
            }
        )
        update_index += 1

        if log_every and episode_returns and len(episode_returns) >= next_console_log:
            next_console_log = len(episode_returns) + log_every
            tail = episode_returns[-20:]
            print(
                f"[ppo {cfg.env_id} seed={cfg.seed}] "
                f"steps {collector.env_steps:6d}  ep {len(episode_returns):4d}  "
                f"mean20={np.mean(tail):7.1f}  KL={kl:.4f}  "
                f"clip={np.mean(clip_fractions) if clip_fractions else 0:.3f}"
                + ("  [early stop on KL]" if stopped_early else ""),
                flush=True,
            )

    collector.close()
    store.insert_policy_updates(update_rows)

    return RunResult(
        experiment_id=experiment_id,
        config=cfg,
        episode_returns=episode_returns,
        episode_steps=episode_steps,
        updates=update_rows,
        actor=actor,
        critic=critic,
    )


def _mean_kl(
    actor: CategoricalActor,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_logp: torch.Tensor,
) -> float:
    """Mean KL(π_old ‖ π_new) over the batch's sampled actions.

    Estimated with the k3 estimator of Schulman's "Approximating KL Divergence".
    Write d = log π_new(a|s) − log π_old(a|s), so that r = exp(d) is the same
    importance ratio the clipped surrogate uses. With the samples drawn from
    π_old, which they were,

        KL(π_old ‖ π_new)  ≈  E[ r − 1 − log r ]  =  E[ exp(d) − 1 − d ]

    rather than the obvious `E[−d]`. Both are unbiased, but `E[−d]` has enormous
    variance and — the part that actually bites — is frequently NEGATIVE on a
    finite batch, which is impossible for a divergence and makes the logged
    column indefensible the first time a reader looks at it. The k3 form is
    non-negative sample by sample, because exp(x) − 1 − x ≥ 0 everywhere.

    Note this is an estimate over the SAMPLED actions, not the exact categorical
    KL over all actions. The exact form is available for a categorical policy
    and would be a defensible alternative; the estimator is used here because it
    is the one PPO implementations report, so your numbers are comparable with
    published ones, and because it is the only form available for the continuous
    policies you will meet in Topic 6.
    """
    with torch.no_grad():
        d = actor.log_prob(obs, actions) - old_logp
        return float((torch.exp(d) - 1.0 - d).mean().item())


def evaluate(actor, env_id: str, episodes: int = 20, seed: int = 12345) -> dict[str, Any]:
    return evaluate_discrete(actor, env_id, episodes, seed)
