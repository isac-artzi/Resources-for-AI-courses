"""
train/policy.py — the parameterised policy pi_theta(a|s), in PyTorch.

TRAINING TIER. This module imports torch and therefore must never be reachable
from `api/` or `ui/`. `tests/test_no_torch.py` asserts that; if you ever import
this file from a serving module, the build fails and it is supposed to.

What changes at Topic 3
-----------------------
Topics 1 and 2 stored a policy as a table: one row per state, and "learning"
meant writing numbers into cells. That does not survive CartPole, whose
observation is four real numbers — there are no cells to write into. So the
policy becomes a FUNCTION APPROXIMATOR: a small network mapping the observation
to a distribution over the two actions, and learning means moving the weights.

Two consequences that the rest of this repository is built around:

  * The policy is stochastic by construction. A softmax output is a probability
    distribution, we sample from it, and the entropy of that distribution is
    what exploration means here. There is no epsilon any more — which is why
    `episodes.epsilon` is null on this topic's runs and `gradient_stats.
    policy_entropy` is the column that replaces it.
  * The deployed artifact is now weights rather than a table, so the export path
    in train/export.py and the NumPy forward pass in api/forward.py stop being
    a formality and become the thing that can silently be wrong. Hence the
    required equivalence test.

Architecture choice: two hidden layers of 64 units with ReLU. Not because 64 is
magic, but because it is the smallest network that reliably solves CartPole,
and a small artifact is a number you are accountable for — this one exports to
roughly 20 KB. Report the size in your model card.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
from torch import nn

DEFAULT_HIDDEN = (64, 64)


def build_mlp(in_dim: int, out_dim: int, hidden: tuple[int, ...] = DEFAULT_HIDDEN) -> nn.Sequential:
    """Linear -> ReLU -> ... -> Linear, with NO activation on the output.

    The missing final activation is deliberate and is the single most common
    source of an equivalence-test failure. This module emits LOGITS; the
    softmax lives in `torch.distributions.Categorical` during training and in
    `api/forward.action_probabilities` at serving time. Appending an
    `nn.Softmax()` here would make PyTorch apply it twice (once in the module,
    once in Categorical) and NumPy apply it once, and the two implementations
    would disagree for a reason that has nothing to do with the export.

    Building it as a flat `nn.Sequential` of Linear/ReLU is also what makes
    `export_torch_mlp` work: it walks `module.modules()` looking for Linear
    layers in order. A fancier module with skip connections would export to
    weights that `api/forward.mlp_forward` cannot reassemble, and it would do so
    without complaining.
    """
    layers: list[nn.Module] = []
    prev = in_dim
    for width in hidden:
        layers += [nn.Linear(prev, width), nn.ReLU()]
        prev = width
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class CategoricalPolicy(nn.Module):
    """pi_theta(a|s) for a discrete action space.

    Everything the policy gradient needs is here and nothing else: the logits,
    a sampler, a log-probability, and an entropy. Notably absent is any notion
    of a return, an advantage or an optimiser — those belong to the algorithm
    (train/vpg.py), not to the policy. Keeping the split means the same policy
    class serves VPG here and A2C/PPO in Topic 4 unchanged.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: tuple[int, ...] = DEFAULT_HIDDEN,
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

    # -- forward -----------------------------------------------------------

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Raw logits. Shape (n_actions,) or (N, n_actions)."""
        return self.net(obs)

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        """The action distribution.

        Constructed from `logits=` and never from `probs=`. `Categorical`
        normalises logits with a log-sum-exp internally, so a logit of 800 —
        which a diverging policy produces — becomes a finite log-probability
        instead of an inf that turns the whole gradient into NaN two steps
        later. Passing hand-computed probabilities gives that stability away.
        """
        return torch.distributions.Categorical(logits=self.forward(obs))

    # -- the three quantities the policy gradient needs ---------------------

    def act(self, obs: np.ndarray) -> tuple[int, float]:
        """SAMPLE an action, and return it with its log-probability.

        Sampling, not argmax, during data collection. The policy gradient
        theorem is an expectation under pi_theta; collecting with a greedy
        policy and updating as though the data came from pi_theta estimates the
        gradient of a different objective, and the run will plateau early for
        no visible reason.

        The log-probability is returned here — under `no_grad`, as a plain float
        — because it is the BEHAVIOUR policy's log pi_old(a|s), the denominator
        of every importance weight computed later. Recomputing it after the
        policy has been updated would give you a ratio of 1 everywhere and an
        importance-sampling arm that silently does nothing.
        """
        with torch.no_grad():
            dist = self.distribution(torch.as_tensor(obs, dtype=torch.float32))
            action = dist.sample()
            return int(action.item()), float(dist.log_prob(action).item())

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """log pi_theta(a|s) for a batch, WITH gradient. This is the estimator."""
        return self.distribution(obs).log_prob(actions)

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """H(pi(.|s)) per state, in nats. Bounded above by ln(n_actions)."""
        return self.distribution(obs).entropy()

    def action_probabilities(self, obs: torch.Tensor) -> torch.Tensor:
        """pi(.|s) as probabilities — the reference the equivalence test compares.

        The test compares this against `api.forward.action_probabilities`. Note
        that it goes through `softmax(logits)` exactly as the NumPy side does,
        so the two are computing the same function on the same weights and any
        difference is either a real bug or float32-versus-float64 rounding.
        """
        with torch.no_grad():
            return torch.softmax(self.forward(obs), dim=-1)

    # -- serialisation -----------------------------------------------------

    def export(self, path: str | pathlib.Path, **extra: np.ndarray) -> dict:
        """Write the deployable `.npz`. See train/export.py for the layout."""
        from train.export import export_torch_mlp

        return export_torch_mlp(self.net, path, **extra)


# ---------------------------------------------------------------------------
# `python -m train.policy --dump-reference DIR --seed 0`
#
# Writes a freshly initialised policy's weights AND its action probabilities on
# a fixed batch of observations into DIR. tests/test_equivalence.py runs this in
# a SUBPROCESS and then compares the saved probabilities against the NumPy
# forward pass in the test process.
#
# Why a subprocess rather than `import torch` at the top of the test: sys.modules
# is per-process, so one torch import anywhere in the suite would leave "torch"
# in sys.modules for every test that runs afterwards — including
# tests/test_no_torch.py, whose entire job is to assert that it is absent. The
# guard that protects the deployment would start failing for a reason that has
# nothing to do with the deployment, and the obvious "fix" is to weaken the
# guard. Paying one subprocess spawn is much cheaper than that.
# ---------------------------------------------------------------------------


def dump_reference(out_dir: str | pathlib.Path, seed: int = 0, n: int = 256) -> dict:
    """Export a seeded policy plus a PyTorch reference batch for the equivalence test."""
    from envs import N_ACTIONS, OBS_DIM

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    policy = CategoricalPolicy(OBS_DIM, N_ACTIONS, seed=seed)
    row = policy.export(out / "equivalence_policy.npz")

    # Deliberately WIDE inputs — N(0, 3) rather than states sampled from a real
    # CartPole rollout, whose four components live in roughly [-2.4, 2.4],
    # [-3, 3], [-0.21, 0.21] and [-3, 3]. A transposed weight matrix can go
    # unnoticed on the narrow, correlated observations a real rollout produces;
    # it cannot hide on inputs that exercise the whole input space, including
    # ones large enough to saturate the softmax.
    rng = np.random.default_rng(seed)
    states = (rng.normal(size=(n, OBS_DIM)) * 3.0).astype(np.float32)
    probs = policy.action_probabilities(torch.as_tensor(states)).numpy()

    np.savez(out / "reference.npz", states=states, probs=probs)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="policy-network utilities (training tier)")
    ap.add_argument("--dump-reference", metavar="DIR", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=256, help="how many reference observations")
    args = ap.parse_args()
    print(json.dumps(dump_reference(args.dump_reference, args.seed, args.n)))


if __name__ == "__main__":  # pragma: no cover - exercised through a subprocess
    main()
