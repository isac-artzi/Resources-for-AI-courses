"""
train/baseline.py — the value network V_phi(s), and the advantage A = R - V(s).

TRAINING TIER (imports torch).

Why a baseline exists at all
----------------------------
The policy gradient estimator is

    g = E[ grad log pi(a|s) * Psi ]

and it is UNBIASED for any Psi of the form (return - b(s)) where b does not
depend on the action. That freedom is the whole trick. Subtracting b(s) leaves
the expected gradient untouched:

    E_a[ grad log pi(a|s) * b(s) ] = b(s) * E_a[ grad log pi(a|s) ]
                                   = b(s) * grad sum_a pi(a|s)
                                   = b(s) * grad 1
                                   = 0

but it changes the VARIANCE, and the variance is what you actually pay for. The
variance-minimising b(s) is close to V(s) — the expected return from s — which
is why the baseline is a value network rather than a constant. Intuitively:
without it, every action taken in a good state gets its probability increased
merely for being taken in a good state. With it, only actions that did better
than the state's own average are reinforced.

The whole argument above is falsifiable, and this product falsifies it: see
`tests/test_baseline_reduces_variance.py` and the Gradient Variance tab.

Note what the baseline is NOT
-----------------------------
It is not a critic in the actor-critic sense. V_phi here is fitted to the
observed Monte Carlo returns and used only to centre them; it never appears in
a bootstrapped target. Replacing R - V(s) with r + gamma*V(s') - V(s) is what
makes this A2C, and that is Topic 4's product, not this one.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from train.policy import DEFAULT_HIDDEN, build_mlp


class ValueNetwork(nn.Module):
    """V_phi(s): observation -> one scalar.

    Same shape as the policy trunk and deliberately so — if the value function
    needs a much larger network than the policy, that is a finding about the
    environment worth reporting, not a hyperparameter to quietly bump.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden: tuple[int, ...] = DEFAULT_HIDDEN,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            # Offset from the policy's seed by the caller, not here. Seeding the
            # value net with the SAME integer as the policy gives two networks
            # whose initial weights are correlated, which is a subtle way to
            # make your seeds less independent than your error bars claim.
            torch.manual_seed(seed)
        self.net = build_mlp(obs_dim, 1, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Shape (N,) — squeezed, so it broadcasts against returns of shape (N,).

        Leaving the trailing dimension on is the classic silent bug here:
        `returns - values` with shapes (N,) and (N, 1) broadcasts to (N, N),
        the MSE is computed over an N-by-N matrix of nonsense, and nothing
        raises. Squeeze at the boundary, once.
        """
        return self.net(obs).squeeze(-1)


def discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    """Reward-to-go: G_t = sum_{k>=t} gamma^(k-t) r_k.

    REWARD-TO-GO, not the whole-episode return repeated at every step. Both are
    unbiased — the rewards collected before time t do not depend on the action
    at time t, so their contribution to the gradient has zero mean — but the
    to-go form has strictly lower variance because it throws away that
    zero-mean noise instead of carrying it. It is the cheapest variance
    reduction available and it costs one backwards loop.

    Computed backwards in one pass. The forward formulation people write first
    is O(T^2) and, on a 500-step CartPole episode, noticeably slow inside a
    training loop.
    """
    out = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def advantages(
    returns: np.ndarray,
    value_net: ValueNetwork | None,
    obs: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """A(s,a) = R - V(s), or plain R when there is no baseline.

    Returns `(advantage, value_estimate)`; the second is logged so the "did the
    value network actually fit anything" question is answerable later from the
    data rather than from memory.

    Two decisions worth stating:

    1. `torch.no_grad()`. V(s) enters the POLICY loss as a constant. If the
       value network's graph were left attached, the policy update would also
       push V around to make the advantage small, which optimises the metric
       instead of the policy. This is the bug that produces a run where the loss
       falls beautifully and the return does not move.

    2. No standardisation. It is standard practice to divide the advantage by
       its batch standard deviation, and `shared/preprocess.normalise_returns`
       is right there. Do NOT use it in this product. Rescaling the advantage
       rescales the gradient, so a normalised no-baseline arm and a normalised
       with-baseline arm would report similar gradient variances no matter what
       the baseline did — you would have normalised away the exact quantity the
       headline chart is measuring. Note it in your README as a deliberate
       deviation from common practice, with this reason.
    """
    if value_net is None:
        return np.asarray(returns, dtype=np.float64), np.zeros_like(returns)
    with torch.no_grad():
        values = value_net(obs).cpu().numpy().astype(np.float64)
    return np.asarray(returns, dtype=np.float64) - values, values


def fit_value_network(
    value_net: ValueNetwork,
    optimiser: torch.optim.Optimizer,
    obs: torch.Tensor,
    returns: np.ndarray,
    epochs: int = 20,
) -> float:
    """Regress V_phi(s) onto the observed returns with MSE. Returns the final loss.

    Fitted AFTER the advantage for this batch has been computed, never before.
    The order is what keeps the baseline honest: a value network fitted on the
    same returns it is about to be subtracted from has already seen the noise in
    them, and subtracting a partly-memorised return removes real signal along
    with the noise. Fit second, use next batch. train/vpg.py enforces the order.

    Several epochs per batch rather than one: the value network is doing plain
    supervised regression, its target is fixed within the batch, and it is
    cheap. Too many, though, and it overfits a batch of ten episodes and gives
    you a baseline that is excellent on data you already have and useless on the
    next batch — watch the loss across batches, not within one.
    """
    target = torch.as_tensor(returns, dtype=torch.float32)
    loss_value = 0.0
    for _ in range(epochs):
        optimiser.zero_grad()
        loss = nn.functional.mse_loss(value_net(obs), target)
        loss.backward()
        optimiser.step()
        loss_value = float(loss.item())
    return loss_value


def explained_variance(returns: np.ndarray, values: np.ndarray) -> float:
    """1 - Var(R - V) / Var(R). The one number that says whether V is doing anything.

    1.0 means V predicts the return perfectly; 0.0 means it is no better than
    predicting the batch mean; NEGATIVE means it is worse than the mean, which
    happens early and is a signal to lower the value learning rate rather than
    to raise it. Quote this in your report next to the variance reduction — a
    baseline that reduces variance while explaining nothing is reducing it by
    being a constant, and you should say so.
    """
    var_r = float(np.var(returns))
    if var_r <= 0.0:
        return 0.0
    return float(1.0 - np.var(returns - values) / var_r)
