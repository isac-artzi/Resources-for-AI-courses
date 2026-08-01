"""
train/nets.py — the networks the three algorithms share, in PyTorch.

TRAINING TIER. This module imports torch and therefore must never be reachable
from `api/` or `ui/`. `tests/test_no_torch.py` asserts that; if you ever import
this file from a serving module, the build fails and it is supposed to.

Why one file for all three algorithms
-------------------------------------
A2C, PPO and SAC differ in their OBJECTIVES, not in their architectures. All
three use a small ReLU trunk; two of them put a softmax on top and one puts a
Gaussian on top. Keeping the networks here and the objectives in a2c.py, ppo.py
and sac.py is what makes the comparison the product exists to make an honest
one: when the three agents share an architecture, a width and an
initialisation scheme, a difference in their learning curves is a difference
between the algorithms rather than between two people's taste in hidden layers.

Architecture: two hidden layers of 64 units with ReLU for the discrete agents,
256 for SAC. Not because those numbers are magic, but because 64 is the smallest
network that reliably solves CartPole and Acrobot, 256 is the width every
published SAC result uses on the classic-control tasks, and a small artifact is
a number you are accountable for — the discrete policies export to roughly 20 KB
and the SAC actor to roughly 280 KB. Report the sizes in your model card.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
from torch import nn

DISCRETE_HIDDEN = (64, 64)
CONTINUOUS_HIDDEN = (256, 256)

# The clamp SAC applies to its log standard deviation. These EXACT numbers are
# written into the exported artifact and re-applied by api/forward.gaussian_head.
# They are part of the function the network computes, not a safety net around
# it: applied on one side only, the two implementations diverge precisely where
# the raw head is extreme.
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def configure_torch_threads(threads: int = 1) -> None:
    """Pin PyTorch to one CPU thread. Called by the CLI entry points, not on import.

    Counter-intuitive but measured: every network in this product is tiny (a
    256×256 matmul at most) and every batch is small, so the work per operation
    is smaller than the cost of waking a thread pool and joining it again. On
    the sandbox this template was verified in, SAC on Pendulum ran roughly 30%
    faster on ONE thread than on all of them — and the multi-threaded run burned
    three times the CPU seconds to do it, which matters when your training job
    is sharing a machine.

    It is called from `main()` in the scripts rather than at import time on
    purpose: a library that silently reconfigures the interpreter's thread pool
    when imported is a library that will surprise somebody. If you move to a
    genuinely large network, delete the call and measure again.
    """
    torch.set_num_threads(max(1, threads))


def build_mlp(in_dim: int, out_dim: int, hidden: tuple[int, ...]) -> nn.Sequential:
    """Linear -> ReLU -> ... -> Linear, with NO activation on the output.

    The missing final activation is deliberate and is the single most common
    source of an equivalence-test failure. These modules emit RAW OUTPUTS:
    logits for the categorical actors (the softmax lives in
    `torch.distributions.Categorical` during training and in
    `api/forward.action_probabilities` at serving time) and a concatenated
    (mean, log σ) for the Gaussian actor. Appending an `nn.Softmax()` or an
    `nn.Tanh()` here would make PyTorch apply it twice and NumPy once, and the
    two implementations would disagree for a reason that has nothing to do with
    the export.

    Building it as a flat `nn.Sequential` of Linear/ReLU is also what makes
    `train/export.py` work: it walks `module.modules()` collecting Linear layers
    in order. A fancier module with skip connections would export to weights
    that `api/forward.mlp_forward` cannot reassemble, and it would do so without
    complaining.
    """
    layers: list[nn.Module] = []
    prev = in_dim
    for width in hidden:
        layers += [nn.Linear(prev, width), nn.ReLU()]
        prev = width
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Discrete: the A2C and PPO actor, and the shared state-value critic
# ---------------------------------------------------------------------------


class CategoricalActor(nn.Module):
    """π_θ(a|s) for a discrete action space. Used unchanged by A2C and PPO.

    Everything the two policy-gradient objectives need is here and nothing else:
    logits, a sampler, a log-probability, an entropy and a probability vector.
    Notably absent is any notion of an advantage, a clip or an optimiser — those
    belong to the algorithm, not to the policy. That split is what lets a2c.py
    and ppo.py be short enough to read side by side, which is the comparison
    this topic is about.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: tuple[int, ...] = DISCRETE_HIDDEN,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            # Seeded HERE, at construction, rather than once at the top of the
            # training script. Weight initialisation is a large part of the
            # seed-to-seed spread you are about to plot as a band, and a run is
            # only reproducible if the initial weights are.
            torch.manual_seed(seed)
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.net = build_mlp(obs_dim, n_actions, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Raw logits. Shape (n_actions,) or (N, n_actions)."""
        return self.net(obs)

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        """The action distribution.

        Constructed from `logits=` and never from `probs=`. `Categorical`
        normalises logits with a log-sum-exp internally, so a logit of 800 —
        which a diverging policy produces — becomes a finite log-probability
        instead of an `inf` that turns the whole gradient into NaN two steps
        later. Passing hand-computed probabilities gives that stability away.
        """
        return torch.distributions.Categorical(logits=self.forward(obs))

    def act(self, obs: np.ndarray) -> tuple[int, float]:
        """SAMPLE an action and return it with its log-probability.

        Sampling, not argmax, during data collection. The policy gradient
        theorem is an expectation under π_θ; collecting with a greedy policy and
        updating as though the data came from π_θ estimates the gradient of a
        different objective, and the run plateaus early for no visible reason.

        The log-probability is captured HERE, under `no_grad`, as a plain float,
        because it is the BEHAVIOUR policy's log π_old(a|s) — the denominator of
        every PPO importance ratio computed later. Recomputing it after the
        policy has been updated gives a ratio of exactly 1 everywhere: a clipped
        surrogate that runs, logs, plots, and does nothing.
        """
        with torch.no_grad():
            dist = self.distribution(torch.as_tensor(obs, dtype=torch.float32))
            action = dist.sample()
            return int(action.item()), float(dist.log_prob(action).item())

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """log π_θ(a|s) for a batch, WITH gradient. This is the estimator."""
        return self.distribution(obs).log_prob(actions)

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """H(π(·|s)) per state, in nats. Bounded above by ln(n_actions)."""
        return self.distribution(obs).entropy()

    def probabilities(self, obs: torch.Tensor) -> torch.Tensor:
        """π(·|s) as probabilities — the reference two equivalence tests compare.

        Goes through `softmax(logits)` exactly as the NumPy side does, so the two
        are computing the same function on the same weights and any difference is
        either a real bug or float32-versus-float64 rounding.
        """
        with torch.no_grad():
            return torch.softmax(self.forward(obs), dim=-1)

    def export(self, path: str | pathlib.Path, env_id: str, **extra) -> dict:
        from train.export import export_torch_mlp

        return export_torch_mlp(self.net, path, env_id=env_id, **extra)


class ValueCritic(nn.Module):
    """V_φ(s) — the critic A2C and PPO share.

    This is the "critic" in actor-critic, and its job is variance reduction, not
    control: it never picks an action. Subtracting V(s) from the return leaves
    the policy gradient unbiased (the proof is in the Concepts tab) and changes
    only its second moment — which is the whole reason actor-critic methods
    learn faster than the pure policy gradient of Topic 3.

    Note that the critic's seed is offset from the actor's by the callers in
    a2c.py and ppo.py. Two networks initialised from the same seed start
    correlated, and a critic that begins life agreeing with the actor is a
    slightly worse baseline than one that does not.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden: tuple[int, ...] = DISCRETE_HIDDEN,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.net = build_mlp(obs_dim, 1, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """V(s), with the trailing singleton dimension squeezed off.

        The squeeze matters. Without it `values` has shape (N, 1), `returns` has
        shape (N,), and `values - returns` broadcasts to (N, N) — an N×N matrix
        of differences whose mean is a perfectly finite number that is not the
        loss you meant. Nothing raises. The symptom is a critic that never fits
        and a memory spike proportional to the square of the batch.
        """
        return self.net(obs).squeeze(-1)


# ---------------------------------------------------------------------------
# Continuous: the SAC actor and its twin critics
# ---------------------------------------------------------------------------


class SquashedGaussianActor(nn.Module):
    """The SAC actor: a state-dependent Gaussian, squashed by tanh, then rescaled.

    Shape of the output layer: `2 * action_dim`. The first half is the mean of a
    Gaussian in PRE-SQUASH space and the second half its log standard deviation.
    Both are functions of the state — which is the point of a maximum-entropy
    actor, and the reason the log σ is a network output here rather than the
    free parameter vector a PPO-continuous implementation would use. An agent
    that must be uncertain near the top of the swing and confident at the bottom
    cannot express that with one number.

    The log-probability correction
    ------------------------------
    Squashing changes the density. If u ~ N(μ, σ) and a = tanh(u), then

        log p(a) = log p(u) − Σ_i log(1 − tanh²(u_i))

    and dropping that second term is the single most common SAC bug: the
    objective still trains, the entropy term is simply wrong, and α — which is
    supposed to trade off against a correct entropy — ends up tuned against a
    quantity that is not the entropy of anything. `sample()` returns the
    corrected log-probability, and the `+ 1e-6` inside the log is there because
    tanh(u)² reaches 1.0 in float32 for |u| ≳ 9, at which point the correction
    is log(0) and the loss is NaN from that step onward.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_scale: float | np.ndarray = 1.0,
        action_bias: float | np.ndarray = 0.0,
        hidden: tuple[int, ...] = CONTINUOUS_HIDDEN,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.net = build_mlp(obs_dim, 2 * act_dim, hidden)
        # Registered as buffers rather than kept as plain floats so they move
        # with `.to(device)` and land in `state_dict()`. They are constants of
        # the environment, not parameters — hence buffers, not Parameters.
        self.register_buffer(
            "action_scale", torch.as_tensor(action_scale, dtype=torch.float32).reshape(-1)
        )
        self.register_buffer(
            "action_bias", torch.as_tensor(action_bias, dtype=torch.float32).reshape(-1)
        )

    def mean_log_std(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(obs)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        # Clamp, not a soft squash. `api/forward.gaussian_head` reproduces this
        # exact `clamp`, and a `tanh`-based bounding here would be a different
        # function that NumPy would not match.
        return mean, torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterised sample plus its exact log-probability.

        `rsample()` rather than `sample()`: SAC differentiates the actor loss
        THROUGH the sampled action into the critic, which requires the sample to
        be a differentiable function of μ and σ (a = μ + σ·ε with ε detached).
        `sample()` returns a leaf with no path back to the parameters, so the
        actor loss would have a zero gradient and the policy would never move —
        silently, with a loss that still goes down because α·log π depends on
        the parameters through log π alone.
        """
        mean, log_std = self.mean_log_std(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        u = normal.rsample()
        squashed = torch.tanh(u)
        log_prob = normal.log_prob(u).sum(-1)
        log_prob = log_prob - torch.log(1.0 - squashed.pow(2) + 1e-6).sum(-1)
        action = self.action_scale * squashed + self.action_bias
        return action, log_prob

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """The MODE of the squashed distribution: `scale * tanh(μ) + bias`.

        Not the mean. tanh is nonlinear, so E[tanh(u)] ≠ tanh(E[u]) and the mean
        of a tanh-squashed Gaussian has no closed form. Serving the mode is the
        standard choice and it is the right one — but it is a choice, and your
        model card should say that the deployed deterministic action is the mode
        rather than an expectation you never computed.
        """
        mean, _ = self.mean_log_std(obs)
        return self.action_scale * torch.tanh(mean) + self.action_bias

    def export(self, path: str | pathlib.Path, env_id: str, **extra) -> dict:
        from train.export import export_squashed_gaussian

        return export_squashed_gaussian(
            self.net,
            path,
            env_id=env_id,
            action_scale=self.action_scale.detach().cpu().numpy(),
            action_bias=self.action_bias.detach().cpu().numpy(),
            log_std_min=LOG_STD_MIN,
            log_std_max=LOG_STD_MAX,
            **extra,
        )


class TwinQ(nn.Module):
    """Two independent Q(s, a) networks, used as `min(Q1, Q2)`.

    Why two. A single Q-network trained with a bootstrapped target
    `r + γ·Q(s', a')` overestimates: the max (or, here, the sampled action from
    an actor trained to maximise Q) selects for exactly the state-action pairs
    where the noise in Q happens to be positive, and the error compounds through
    the bootstrap. Taking the minimum of two independently initialised critics
    biases the estimate DOWNWARD, which is the cheap and effective fix that
    TD3 introduced and SAC adopted. An underestimate is a much safer error than
    an overestimate here, because the actor is optimising against this number.

    The critics stay in the training tier. They are never exported: the deployed
    artifact is the ACTOR, and a Q-network is how you trained it, not part of
    what you deployed.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: tuple[int, ...] = CONTINUOUS_HIDDEN,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.q1 = build_mlp(obs_dim + act_dim, 1, hidden)
        self.q2 = build_mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([obs, act], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)


# ---------------------------------------------------------------------------
# Advantage estimation, shared by A2C and PPO
# ---------------------------------------------------------------------------


def generalised_advantage(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    truncateds: np.ndarray,
    last_value: float,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE(λ). Returns (advantages, value targets).

    The recursion, backwards through the batch:

        δ_t = r_t + γ·V(s_{t+1})·[not terminal] − V(s_t)
        A_t = δ_t + γ·λ·A_{t+1}·[not episode boundary]

    Two details that are the difference between this working and not:

    **`dones` means TERMINATED, not "the episode ended".** A truncated episode —
    CartPole hitting its 500-step limit, Pendulum hitting 200 — has a future
    worth V(s'), and zeroing that bootstrap teaches the critic that surviving to
    the time limit is exactly as bad as failing. That is why `truncateds` is a
    separate argument rather than being folded into `dones`: the bootstrap uses
    `dones`, the advantage recursion resets at either.

    **λ is a bias/variance dial, not a constant to copy.** λ = 1 recovers the
    Monte Carlo advantage (unbiased, high variance — Topic 3's estimator); λ = 0
    recovers the one-step TD advantage (low variance, biased by however wrong
    the critic is). 0.95 is the usual compromise and is what this product uses;
    it is a hyperparameter, and it is in `experiments.hyperparameters` so you can
    say which value produced which curve.
    """
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float64)
    running = 0.0
    next_value = float(last_value)
    for t in range(n - 1, -1, -1):
        # The bootstrap is zeroed only on a TERMINAL state.
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        # The recursion resets at ANY episode boundary — a new episode's
        # advantage cannot depend on the previous one's.
        boundary = 1.0 if (dones[t] or truncateds[t]) else 0.0
        running = delta + gamma * lam * running * (1.0 - boundary)
        adv[t] = running
        next_value = values[t]
    # Value targets are advantage + value, which is the λ-return. Fitting the
    # critic on this rather than on the Monte Carlo return keeps the two
    # quantities consistent: the advantage the actor consumed and the target the
    # critic chased are the same estimate of the same thing.
    return adv, adv + values


# ---------------------------------------------------------------------------
# `python -m train.nets --dump-reference DIR --seed 0`
#
# Writes three freshly initialised policies — one per environment, matching the
# three deployed artifacts' shapes — plus their PyTorch outputs on a fixed batch
# of observations. tests/test_equivalence.py runs this in a SUBPROCESS and
# compares the saved outputs against the NumPy forward pass in the test process.
#
# Why a subprocess rather than `import torch` at the top of the test: sys.modules
# is per-process, so one torch import anywhere in the suite would leave "torch"
# in sys.modules for every test that runs afterwards — including
# tests/test_no_torch.py, whose entire job is to assert that it is absent. The
# guard that protects the deployment would start failing for a reason that has
# nothing to do with the deployment, pytest collection order would decide
# whether the build was green, and the obvious "fix" would be to weaken the
# guard. Paying one interpreter spawn is much cheaper than that.
# ---------------------------------------------------------------------------


def dump_reference(out_dir: str | pathlib.Path, seed: int = 0, n: int = 256) -> dict:
    """Export one policy per environment plus a PyTorch reference batch for each.

    The observations are deliberately WIDE — N(0, 3) rather than states sampled
    from a real rollout, whose components live in narrow, correlated ranges. A
    transposed weight matrix can go unnoticed on the observations a real rollout
    produces; it cannot hide on inputs that exercise the whole input space,
    including ones large enough to saturate a softmax and to drive the SAC log σ
    into its clamp — which is exactly where a clamp applied on one side only
    would show up.
    """
    from envs import ENV_SPECS

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest: dict[str, dict] = {}

    for env_id, spec in ENV_SPECS.items():
        states = (rng.normal(size=(n, spec.obs_dim)) * 3.0).astype(np.float32)
        stem = env_id.split("-")[0].lower()
        npz = out / f"equivalence_{stem}.npz"

        if spec.discrete:
            actor = CategoricalActor(spec.obs_dim, spec.n_actions, seed=seed)
            row = actor.export(npz, env_id=env_id)
            probs = actor.probabilities(torch.as_tensor(states)).numpy()
            np.savez(out / f"reference_{stem}.npz", states=states, probs=probs)
        else:
            actor = SquashedGaussianActor(
                spec.obs_dim,
                spec.n_actions,
                action_scale=spec.action_high,
                action_bias=0.0,
                seed=seed,
            )
            row = actor.export(npz, env_id=env_id)
            with torch.no_grad():
                t = torch.as_tensor(states)
                mean, log_std = actor.mean_log_std(t)
                action = actor.deterministic_action(t)
            np.savez(
                out / f"reference_{stem}.npz",
                states=states,
                mean=mean.numpy(),
                log_std=log_std.numpy(),
                action=action.numpy(),
            )
        manifest[env_id] = row

    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="network utilities (training tier)")
    ap.add_argument("--dump-reference", metavar="DIR", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=256, help="how many reference observations")
    args = ap.parse_args()
    print(json.dumps(dump_reference(args.dump_reference, args.seed, args.n)))


if __name__ == "__main__":  # pragma: no cover - exercised through a subprocess
    main()
