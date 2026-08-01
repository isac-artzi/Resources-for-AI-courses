"""
The claim this product exists to defend, as a test.

The theory in train/baseline.py says that subtracting a state-dependent
baseline leaves the policy gradient unbiased and reduces its variance. That is a
claim about YOUR code, not about the literature, and the difference between the
two is the whole point of the product brief: your team lead wants evidence, not
adjectives.

So: on a fixed seed, with the SAME trajectories and the SAME policy parameters,
measure the trace of the covariance of the per-episode gradient estimates with
and without the baseline, and assert that the baseline reduces it.

Why this is a fair test and the obvious version is not
-----------------------------------------------------
The tempting version — train two agents, compare their logged gradient
variance — is confounded twice over. Different runs see different trajectories,
so you measure the seed; and in CartPole the return IS the episode length, so an
arm that learns faster has larger returns, more terms in the sum, and therefore
larger raw gradient variance for a reason that has nothing to do with the
baseline. Held-fixed trajectories and held-fixed parameters remove both. See
`train.vpg.compare_baseline_variance`.

Runs in a subprocess because it needs torch and this process must not have it —
see `run_torch_script` in conftest.py.
"""

from __future__ import annotations

import pytest

# Three seeds, not one. A variance ratio is itself a random variable and a
# single draw of it is an anecdote; if the effect were marginal, one seed would
# find it half the time. It is not marginal — the observed ratios are 3x to 6x —
# and requiring all three to clear the bar is what makes a regression visible.
SEEDS = (0, 1, 2)

# The bar is "strictly reduces", plus a margin so a 1% wobble does not turn the
# suite red. It is NOT set at the observed 3x: a threshold tuned to today's
# measurement is a test that fails when someone legitimately changes the network
# width, and the theoretical claim is about the sign, not the size.
MIN_RATIO = 1.2


@pytest.fixture(scope="module")
def measurements(requires_torch):
    from tests.conftest import run_torch_script

    return run_torch_script(
        f"""
        from train.vpg import compare_baseline_variance
        out = {{}}
        for s in {SEEDS!r}:
            out[str(s)] = compare_baseline_variance(seed=s)
        print(json.dumps(out))
        """
    )


def test_baseline_reduces_gradient_variance_on_every_seed(measurements):
    for seed, m in measurements.items():
        ratio = m["variance_ratio"]
        assert ratio > MIN_RATIO, (
            f"seed {seed}: the baseline did NOT reduce gradient variance "
            f"(without={m['without_baseline_variance']:.4g}, "
            f"with={m['with_baseline_variance']:.4g}, ratio={ratio:.3f}).\n"
            "Do not relax this threshold. Check, in order:\n"
            "  * that the advantage is R - V(s) and not V(s) - R (a sign error "
            "    here increases variance and still trains, badly);\n"
            "  * that V is evaluated under torch.no_grad() so it enters the "
            "    policy loss as a constant;\n"
            "  * that nothing standardises the advantage — dividing by its batch "
            "    standard deviation rescales BOTH arms to a similar variance and "
            "    erases the effect being measured (see train/baseline.py);\n"
            "  * that the value network actually fitted: explained_variance "
            f"    here is {m['with_baseline_explained_variance']:.3f} and should "
            "    be well above 0."
        )
        print(
            f"seed {seed}: variance {m['without_baseline_variance']:.4g} -> "
            f"{m['with_baseline_variance']:.4g}  (x{ratio:.2f} reduction, "
            f"explained variance {m['with_baseline_explained_variance']:.3f})"
        )


def test_the_value_network_explains_a_real_share_of_the_return(measurements):
    """A baseline can reduce variance by being a constant. Say which one yours is.

    If explained variance were near zero while the ratio were large, the value
    network would be doing nothing that the batch mean could not do, and the
    honest description of your method would be "mean-centred returns" rather
    than "a learned value baseline". That is a defensible method — it is just a
    different one from the one your model card claims.
    """
    for seed, m in measurements.items():
        ev = m["with_baseline_explained_variance"]
        assert ev > 0.3, (
            f"seed {seed}: V(s) explains only {ev:.3f} of the variance in the "
            "returns. Raise value_epochs or value_lr, or check that "
            "ValueNetwork.forward squeezes its output — an unsqueezed (N, 1) "
            "against an (N,) target broadcasts to (N, N) and trains on nonsense "
            "without raising."
        )


def test_the_baseline_is_not_secretly_changing_the_gradient_direction(measurements):
    """Unbiasedness, checked the only way a finite sample can check it.

    The theory says E[g] is the same with and without a baseline; a finite batch
    cannot prove that, but it CAN catch the failure that matters — a baseline
    that has been wired in with a sign error or applied to the wrong states will
    produce a mean gradient pointing somewhere unrelated, not merely a noisier
    one. Both norms being finite and non-degenerate is the weak check that fits
    in a unit test; the strong check is that both arms in the ablation learn.
    """
    for seed, m in measurements.items():
        assert m["with_baseline_norm"] > 0.0, f"seed {seed}: zero gradient with a baseline"
        assert m["without_baseline_norm"] > 0.0
        assert m["without_baseline_explained_variance"] == 0.0, (
            "with no baseline the 'value estimate' must be exactly zero, so it "
            "explains exactly none of the return's variance; anything else means "
            "the no-baseline arm is quietly using one"
        )
