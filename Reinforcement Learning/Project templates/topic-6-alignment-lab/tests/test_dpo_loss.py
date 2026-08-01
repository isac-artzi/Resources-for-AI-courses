"""
The DPO loss behaves the way the derivation says it does.

DQ 3(b) asks for the loss in fewer than fifteen lines of PyTorch and for a
numerical check that it decreases as the chosen-response margin increases.
That check is here rather than in a notebook cell, because the bug it catches
is silent: a DPO implementation with the two log-ratios swapped trains happily,
reports a falling loss, and produces a policy that has learned to prefer the
REJECTED response. Nothing downstream notices — the KL still rises, the
completions still change, and the reward model still scores them.

SUBPROCESS, NOT IMPORT. `train.dpo` needs torch, and this test process must
stay free of it or `tests/test_no_torch.py` becomes a lie about the deployment
budget. See `run_torch_script` in conftest.py for the full argument.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def loss_curves(request):
    request.getfixturevalue("requires_torch")
    from tests.conftest import run_torch_script

    return run_torch_script(
        """
        from train.dpo import verify_loss_decreases_with_margin
        print(json.dumps(verify_loss_decreases_with_margin((0.05, 0.1, 0.5))))
        """
    )


def test_loss_decreases_monotonically_as_the_margin_grows(loss_curves):
    """The sign check. This is the assertion that catches a swapped log-ratio."""
    margins = np.asarray(loss_curves["margins"])
    for key in [k for k in loss_curves if k.startswith("beta=")]:
        losses = np.asarray(loss_curves[key])
        diffs = np.diff(losses)
        assert np.all(diffs < 0), (
            f"the DPO loss did not decrease monotonically in the margin at {key}.\n"
            f"margins: {margins.tolist()}\nlosses:  {losses.tolist()}\n"
            "The usual cause is the two log-ratios swapped in dpo_loss — the "
            "policy would then be trained to prefer the REJECTED response, and "
            "the training curve would look completely normal."
        )
    assert loss_curves["monotone_decreasing"] is True


def test_a_zero_margin_gives_the_coin_flip_loss(loss_curves):
    """At margin zero the loss must be -log sigma(0) = log 2 = 0.6931, at EVERY beta.

    A useful invariant because it is beta-independent: `beta * 0 = 0` whatever
    beta is. A loss at margin zero that varies with beta means beta has been
    applied somewhere it should not be — inside the log, or to the reference
    log-probabilities rather than to the ratio.
    """
    margins = np.asarray(loss_curves["margins"])
    zero = int(np.argmin(np.abs(margins)))
    assert abs(margins[zero]) < 1e-9, "the sweep should contain margin exactly 0"
    for key in [k for k in loss_curves if k.startswith("beta=")]:
        assert loss_curves[key][zero] == pytest.approx(np.log(2.0), abs=1e-6), (
            f"{key}: loss at margin 0 is {loss_curves[key][zero]}, expected log 2. "
            "beta must multiply the log-ratio difference, nothing else."
        )


def test_larger_beta_sharpens_the_loss(loss_curves):
    """Beta is the inverse temperature of the implicit reward.

    At a POSITIVE margin, a larger beta pushes `sigma(beta * margin)` closer to
    1 and the loss closer to 0; at a negative margin it does the opposite. So
    the loss curves for different betas must cross at margin 0 and nowhere
    else. If they do not, beta is being applied asymmetrically.
    """
    margins = np.asarray(loss_curves["margins"])
    small = np.asarray(loss_curves["beta=0.05"])
    large = np.asarray(loss_curves["beta=0.5"])
    pos, neg = margins > 1e-9, margins < -1e-9
    assert np.all(large[pos] < small[pos]), "a larger beta must lower the loss on a won pair"
    assert np.all(large[neg] > small[neg]), "a larger beta must raise the loss on a lost pair"


def test_the_alignment_run_actually_moved_the_policy(request):
    """A short end-to-end run: KL must be positive and fall as beta rises.

    Two assertions, and the second is the one with teeth. A positive KL only
    says the policy moved. KL DECREASING IN BETA is the statement that the KL
    coefficient is doing what it is named for, and it is the property the whole
    reward-hacking sweep rests on. If it does not hold, every chart downstream
    is plotting a variable that is not controlling anything.
    """
    request.getfixturevalue("requires_torch")
    from tests.conftest import run_torch_script

    out = run_torch_script(
        """
        from shared.store import MemoryStore
        from train.data import load_synthetic
        from train.dpo import run_alignment
        ds = load_synthetic(n_pairs=600, seed=0)
        s = run_alignment(ds, betas=(0.05, 0.5), sft_steps=120, dpo_steps=150,
                          n_gen_prompts=12, seed=0, store=MemoryStore(),
                          score_fn=lambda t: 0.0)
        print(json.dumps({"runs": [{k: r[k] for k in
              ("beta", "kl_from_reference", "implicit_reward_accuracy",
               "implicit_reward_margin")} for r in s["runs"]]}))
        """,
        timeout=900,
    )
    runs = {r["beta"]: r for r in out["runs"]}
    for beta, r in runs.items():
        assert r["kl_from_reference"] > 0.0, (
            f"beta={beta}: KL from the reference is {r['kl_from_reference']}. "
            "A KL of exactly zero means the policy did not move — the usual "
            "cause is a reference that shares parameters with the policy "
            "instead of being a frozen deepcopy."
        )
        assert 0.0 <= r["implicit_reward_accuracy"] <= 1.0
    assert runs[0.5]["kl_from_reference"] < runs[0.05]["kl_from_reference"], (
        f"KL at beta=0.5 ({runs[0.5]['kl_from_reference']:.2f}) is not below KL at "
        f"beta=0.05 ({runs[0.05]['kl_from_reference']:.2f}). A larger beta is a "
        "TIGHTER constraint and must keep the policy closer to the reference."
    )
