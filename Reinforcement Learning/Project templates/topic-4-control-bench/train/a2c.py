"""
train/a2c.py — advantage actor-critic on CartPole-v1.

TRAINING TIER (imports torch). Never reachable from api/ or ui/.

A2C is the smallest thing in this topic and the right place to start reading,
because the other two are both A2C with one idea added:

    A2C  = policy gradient + a learned critic                  (this file)
    PPO  = A2C + a clipped ratio so a batch can be reused      (train/ppo.py)
    SAC  = A2C's structure, off-policy, with entropy in the
           objective rather than as a bonus on the loss        (train/sac.py)

The loop
--------
    1. Run the current policy for `n_steps` environment steps.
    2. Estimate the advantage with GAE(λ), bootstrapping the value of wherever
       we stopped.
    3. Take ONE gradient step on
           −E[ log π(a|s)·A ]  −  c_H·E[H(π)]  +  c_V·E[(V(s) − target)²]
    4. Write one `episodes` row per finished episode and one `policy_updates`
       row per update.

Step 3 is the whole difference from Topic 3. There, the weight on
`log π(a|s)` was the Monte Carlo return minus a baseline fitted to it. Here it
is a λ-return minus the critic's estimate, and the critic is updated in the same
step. The gain is variance: the critic replaces a sum of hundreds of noisy
rewards with one number, at the cost of whatever bias the critic carries. That
trade is the topic.

**One update per batch, and that is what makes A2C A2C.** PPO takes ten epochs
over the same data; A2C takes one step and throws the batch away. It is
therefore strictly on-policy, needs no importance ratio, and needs no trust
region — the policy cannot drift away from the data it was collected under
because it only sees that data once. It is also, for the same reason,
sample-inefficient, and that is precisely the comparison `train/compare.py`
measures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from envs import spec_for
from train.nets import DISCRETE_HIDDEN, CategoricalActor, ValueCritic, generalised_advantage
from train.onpolicy import StepCollector, episode_rows, evaluate_discrete


@dataclass
class A2CConfig:
    """Every knob, in one place, and every one of them lands in
    `experiments.hyperparameters` as jsonb.

    That is not bookkeeping. Six weeks from now the only way to answer "what was
    different about run 7" is to read the row, and a hyperparameter that lived in
    a command-line default rather than in the database is a hyperparameter you
    will not recover.
    """

    env_id: str = "CartPole-v1"
    total_steps: int = 60_000
    seed: int = 0

    # 128 steps per update. Short on purpose: A2C takes ONE gradient step per
    # batch, so the batch length and the number of updates are the same dial
    # seen from two ends. At 60,000 steps this is 469 updates; at 1,024 steps
    # per batch it would be 58, which is not enough for any learning rate to
    # move a randomly initialised policy to a solved CartPole. If your A2C looks
    # like it is not learning, count your updates before you touch the learning
    # rate.
    n_steps: int = 128

    gamma: float = 0.99
    # GAE λ. 1.0 recovers the Monte Carlo advantage of Topic 3 (unbiased, high
    # variance); 0.0 recovers one-step TD (low variance, biased by however wrong
    # the critic is). 0.95 is the usual compromise.
    gae_lambda: float = 0.95

    # 1e-3 with Adam. High by supervised standards and normal for a policy
    # gradient with a few hundred updates: a learning rate tuned for supervised
    # training (1e-4, say) leaves the policy essentially where it started and
    # looks exactly like an implementation bug.
    #
    # It is also the value this template SETTLED ON rather than started with.
    # 3e-3 learns visibly faster early and then oscillates: on seed 0 at 60,000
    # steps it reached a training mean of 141 and a GREEDY evaluation of 98,
    # which is the signature of a policy that keeps overshooting and recovering.
    # 1e-3 reached 211 and 500 on the same seed. That is what a policy-gradient
    # learning rate that is too high looks like — not divergence, just a curve
    # that stops improving and a deployed policy that is worse than the training
    # curve suggests. Check the greedy evaluation, not only the curve.
    policy_lr: float = 1e-3
    value_lr: float = 1e-3
    hidden: tuple[int, ...] = DISCRETE_HIDDEN

    # The entropy bonus. NOT the same thing as SAC's α, and the difference is a
    # discussion question: this term is added to the LOSS as a regulariser and
    # decays in importance as the policy sharpens, whereas SAC puts entropy
    # inside the REWARD, so the optimal policy itself is stochastic. Same
    # symbol, different objective. 0.01 is enough to stop CartPole collapsing to
    # a deterministic policy in the first hundred updates.
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    # Global-norm clipping on the shared backward pass. A single unlucky batch
    # in a policy gradient can produce a gradient two orders of magnitude larger
    # than the median one, and one such step can destroy a policy that took
    # 300 updates to build. 0.5 is the standard value.
    max_grad_norm: float = 0.5

    # Advantage standardisation, within the batch. Note that Topic 3 deliberately
    # did NOT do this, because that product was measuring gradient variance and
    # normalising the advantage would have normalised away the quantity being
    # measured. This product measures returns and sample efficiency, where the
    # scale-freedom is worth having: it makes one learning rate work across
    # CartPole's +1-per-step rewards and Acrobot's −1-per-step ones.
    normalise_advantages: bool = True

    @property
    def algorithm(self) -> str:
        return "a2c"

    def as_hyperparameters(self) -> dict[str, Any]:
        d = asdict(self)
        d["hidden"] = list(self.hidden)  # jsonb has no tuple type
        return d


@dataclass
class RunResult:
    experiment_id: str
    config: Any
    episode_returns: list[float]
    episode_steps: list[int]
    updates: list[dict[str, Any]]
    actor: Any
    critic: Any

    def mean_return_last(self, n: int = 100) -> float:
        """The headline learning number. Quote the window with it, always.

        "Mean return 180" is not a claim; "mean return over the last 100 of 412
        episodes, seed 0" is. A method rather than a stored field because the
        window is an argument to the claim, not a property of the run.
        """
        tail = self.episode_returns[-n:]
        return float(np.mean(tail)) if tail else 0.0

    def episodes_to_threshold(self, threshold: float, window: int = 10) -> int | None:
        """First episode whose trailing `window` mean reached `threshold`, or None.

        None rather than a sentinel. A run that never got there is a result, and
        encoding it as 9999 puts a number into the mean of the column that is
        not a measurement of anything.
        """
        r = self.episode_returns
        for i in range(window - 1, len(r)):
            if float(np.mean(r[i - window + 1 : i + 1])) >= threshold:
                return i
        return None


def train_a2c(cfg: A2CConfig, store: Any | None = None, log_every: int = 20) -> RunResult:
    """Train one A2C run and write every row it is supposed to write.

    The product brief says "log every episode", and this does: one `episodes`
    row per completed episode, written as the run proceeds rather than at the
    end. That ordering is not incidental — a run that crashes at step 55,000 of
    60,000 should leave the episodes it completed behind, and a training script
    that buffers everything in memory and writes once leaves nothing.

    `log_every` is a CONSOLE setting, in episodes. It has nothing to do with what
    is stored.
    """
    from shared.store import get_store

    store = store or get_store()
    spec = spec_for(cfg.env_id)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    actor = CategoricalActor(spec.obs_dim, spec.n_actions, cfg.hidden, seed=cfg.seed)
    # The critic's seed is OFFSET from the actor's. Two networks initialised
    # from the same seed start correlated, and a critic that begins life
    # agreeing with the actor is a slightly worse baseline than one that does
    # not. The offset costs nothing and removes the question.
    critic = ValueCritic(spec.obs_dim, cfg.hidden, seed=cfg.seed + 10_000)

    # Two optimisers rather than one over the union of the parameters. The actor
    # and the critic are solving different problems — one ascends a policy
    # gradient, the other descends a regression loss — and a single optimiser
    # forces them to share a learning rate and, with Adam, a single set of
    # moment estimates. Separate optimisers is also what lets you tune one
    # without touching the other, which is the first thing you will want to do.
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
    # Every episode becomes a ROW; `log_every` only controls how often one is
    # also printed. Console output is for you, the database is the deliverable,
    # and conflating the two is how a run ends up with a beautiful terminal and
    # an empty `episodes` table.
    next_console_log = 0

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

        logp = actor.log_prob(obs_t, act_t)
        entropy = actor.entropy(obs_t).mean()
        # `-` because optimisers descend. Ascending a gradient by descending its
        # negation is the whole of "gradient ascent" in PyTorch, and writing it
        # out is cheaper than the sign error you get from remembering it.
        policy_loss = -(logp * adv_t).mean() - cfg.entropy_coef * entropy

        value_pred = critic(obs_t)
        value_loss = torch.nn.functional.mse_loss(value_pred, tgt_t)

        actor_opt.zero_grad(set_to_none=True)
        critic_opt.zero_grad(set_to_none=True)
        (policy_loss + cfg.value_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
        actor_opt.step()
        critic_opt.step()

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
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "policy_entropy": float(entropy.item()),
                # NULL, and honestly so. A2C takes ONE step per batch and never
                # asks how far it moved; there is no old policy to measure
                # against because the batch is discarded immediately. Writing
                # 0.0 here would make A2C look like the most conservative method
                # in the study when in fact it is the one that does not look.
                "kl_divergence": None,
                "clip_fraction": None,
                "alpha": None,
            }
        )
        update_index += 1

        if log_every and episode_returns and len(episode_returns) >= next_console_log:
            next_console_log = len(episode_returns) + log_every
            tail = episode_returns[-20:]
            print(
                f"[a2c {cfg.env_id} seed={cfg.seed}] "
                f"steps {collector.env_steps:6d}  ep {len(episode_returns):4d}  "
                f"mean20={np.mean(tail):7.1f}  H={float(entropy.item()):.3f}  "
                f"vloss={float(value_loss.item()):.3f}",
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


def evaluate(actor, env_id: str, episodes: int = 20, seed: int = 12345) -> dict[str, Any]:
    return evaluate_discrete(actor, env_id, episodes, seed)
