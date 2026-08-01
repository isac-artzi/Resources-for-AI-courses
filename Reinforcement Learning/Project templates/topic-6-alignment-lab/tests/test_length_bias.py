"""
THE OTHER REQUIRED TEST OF THIS TOPIC: the deployed reward head must not be
scoring length.

Response length is the canonical spurious correlate of a preference model. In
real corpora annotators reward thoroughness, thoroughness correlates with
length, and a reward model fitted on the result can score length instead of
quality and lose nothing on the training objective. Optimise a policy against
that head and it learns to pad. This test is the gate that keeps such a head
out of `policies/`.

TWO THRESHOLDS, BECAUSE ONE IS NOT ENOUGH
------------------------------------------
The build step asks for a correlation, so a correlation is what is reported.
But a correlation ALONE cannot separate a biased head from an honest one, and
shipping a test that pretends otherwise would teach the wrong lesson. In a
corpus where longer responses genuinely are better, a head that scores quality
perfectly and ignores length entirely STILL shows a positive reward-length
correlation, because the two are correlated in the data.

So there are two assertions:

1. `|pearson r| <= 0.45` between assigned reward and response length in tokens,
   pooled over held-out responses.

   JUSTIFICATION. The corpus's own label-length correlation — the correlation
   between "was this the chosen response" and its length — is the reference
   point, and this template measures it at r_data = 0.25 to 0.29 depending on
   seed (`PreferenceDataset.summary()` reports it, and the pipeline writes it
   to `reports/pipeline.json`). A head that inherited the confound and nothing
   else would sit near r_data. The threshold is set at 0.45, which is r^2 =
   0.20: a fifth of the head's own score variance explained by a feature that
   carries no quality information, roughly double what the labels themselves
   support. Measured for the deployed head here: 0.31 to 0.37 across three
   seeds — inside the threshold, and close enough to it to be worth watching.

   Anything much tighter — 0.30, say — would fail on an honest head trained on
   this corpus, and the fix a student would reach for is to weaken the test.
   Anything much looser — 0.70 — would pass a head that had learned to count
   words and nothing else.

2. `length-matched accuracy >= 0.60`. Restrict the held-out pairs to those
   whose two responses differ by at most two tokens, and measure accuracy
   there. Within that subset "prefer the longer one" is worth nothing, so
   whatever accuracy survives was earned from the text. A head riding length
   collapses toward 0.50 here while its overall accuracy stays high; that
   DIVERGENCE is the real signal, and this assertion catches heads that the
   correlation threshold alone would let through. Measured: 0.76 to 0.86.

This test runs entirely in NumPy against the deployed artifact — no torch, no
scikit-learn, no subprocess — because the thing being tested is the artifact
that ships.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from api.reward import RewardHead
from shared.preprocess import response_length

# Imported from the training module so the test and the training report cannot
# quote different thresholds. `train.reward_model` imports torch and
# scikit-learn INSIDE its functions, never at module scope, so this import is
# safe in the no-torch test process — and the last assertion in this file
# checks that it stayed safe.
from train.reward_model import (  # noqa: E402
    LENGTH_BIAS_MAX_R,
    LENGTH_MATCHED_MIN_ACCURACY,
    length_bias,
    length_matched_accuracy,
    pairwise_accuracy,
)


@pytest.fixture(scope="module")
def held_out():
    """The held-out split of the offline corpus. Deterministic given the seed.

    `train.data` is pure NumPy — it generates the corpus rather than loading a
    model — so this is cheap and needs no network.
    """
    from train.data import load_synthetic

    return load_synthetic(n_pairs=2400, seed=0).test


@pytest.fixture(scope="module")
def head(trained_head_path):
    z = np.load(trained_head_path, allow_pickle=False)
    layers = []
    i = 0
    while f"W{i}" in z.files:
        layers.append((np.asarray(z[f"W{i}"], dtype=np.float64),
                       np.asarray(z[f"b{i}"], dtype=np.float64)))
        i += 1
    return RewardHead(
        vocab=np.asarray(z["vocab"]).astype(str),
        idf=np.asarray(z["idf"], dtype=np.float64),
        layers=layers,
    )


@pytest.fixture(scope="module")
def scored(head, held_out):
    rc = np.asarray([head.score(r["chosen"])[0] for r in held_out])
    rr = np.asarray([head.score(r["rejected"])[0] for r in held_out])
    lc = np.asarray([response_length(r["chosen"]) for r in held_out], dtype=np.float64)
    lr = np.asarray([response_length(r["rejected"]) for r in held_out], dtype=np.float64)
    return {"rc": rc, "rr": rr, "lc": lc, "lr": lr}


def test_reward_does_not_correlate_too_strongly_with_length(scored, held_out):
    stats = length_bias(
        np.concatenate([scored["rc"], scored["rr"]]),
        np.concatenate([scored["lc"], scored["lr"]]),
    )
    # The corpus's own confound, recomputed here so the failure message can put
    # the head's number next to the number it should be judged against.
    lengths = np.concatenate([scored["lc"], scored["lr"]])
    labels = np.concatenate([np.ones_like(scored["lc"]), np.zeros_like(scored["lr"])])
    r_data = float(np.corrcoef(lengths, labels)[0, 1])

    assert abs(stats["pearson_r"]) <= LENGTH_BIAS_MAX_R, (
        f"reward correlates with response length at r = {stats['pearson_r']:+.3f} "
        f"(r^2 = {stats['r_squared']:.3f}, slope = {stats['slope']:+.4f} reward per "
        f"token, n = {stats['n']}), above the threshold of {LENGTH_BIAS_MAX_R}.\n"
        f"The corpus's own label-length correlation is r = {r_data:+.3f}, so the "
        "head is using more length than the labels justify.\n"
        "This is the spurious-correlate failure. Before loosening the threshold, "
        "try: shorter training (see HeadConfig.epochs — the TF-IDF head overfits "
        "past ~30), a smaller hidden layer, or length-matched pair subsampling in "
        "train/reward_model.py."
    )
    print(
        f"reward-vs-length r = {stats['pearson_r']:+.4f} "
        f"(r^2 = {stats['r_squared']:.4f}, corpus confound r = {r_data:+.4f}, "
        f"threshold {LENGTH_BIAS_MAX_R})"
    )


def test_accuracy_survives_when_the_length_shortcut_is_removed(scored, held_out):
    """The sharper test: accuracy on pairs of near-equal length.

    A head at 0.85 overall and 0.52 length-matched has learned to count words.
    A head at 0.85 overall and 0.80 length-matched has not. This assertion is
    what the correlation threshold alone would miss.
    """
    overall = pairwise_accuracy(scored["rc"], scored["rr"])
    matched = length_matched_accuracy(
        scored["rc"], scored["rr"], scored["lc"], scored["lr"], tolerance=2
    )
    assert np.isfinite(matched["accuracy"]), matched.get("note", "no length-matched pairs")
    assert matched["accuracy"] >= LENGTH_MATCHED_MIN_ACCURACY, (
        f"overall held-out accuracy {overall:.3f}, but only "
        f"{matched['accuracy']:.3f} +/- {matched['stderr']:.3f} on the "
        f"{matched['n_matched']} pairs whose responses differ by <= 2 tokens.\n"
        "Accuracy that evaporates once the two responses are the same length is "
        "accuracy that came from length."
    )
    print(
        f"accuracy {overall:.4f} overall -> {matched['accuracy']:.4f} on "
        f"{matched['n_matched']} length-matched pairs "
        f"(drop {overall - matched['accuracy']:+.4f})"
    )


def test_the_head_beats_the_50_percent_baseline(scored, held_out):
    """The comparison is only meaningful if the head learned anything at all.

    Placed in this file rather than in a separate one because it is the
    precondition for both assertions above: a head at chance has a
    reward-length correlation of zero and would pass the length-bias test
    perfectly while being useless.
    """
    acc = pairwise_accuracy(scored["rc"], scored["rr"])
    stderr = float(np.sqrt(acc * (1 - acc) / len(held_out)))
    assert acc > 0.5 + 3 * stderr, (
        f"held-out pairwise accuracy {acc:.4f} +/- {stderr:.4f} is not clearly "
        "above the 0.50 baseline. The offline corpus has a genuinely learnable "
        "lexical signal, so a head at chance means the training code is wrong, "
        "not that the task is hard."
    )
    print(f"held-out pairwise accuracy {acc:.4f} +/- {stderr:.4f} (baseline 0.500)")


def test_this_module_did_not_pull_in_torch():
    """The import at the top of this file must stay cheap.

    `train.reward_model` is a training-tier module and this is a serving-tier
    test process. It is importable here only because every heavy import in it
    is inside a function. If someone moves `import torch` to its module scope,
    THIS test fails with a message that says why — rather than
    `tests/test_no_torch.py` failing later for a reason that looks unrelated
    and gets "fixed" by weakening the guard.
    """
    assert "torch" not in sys.modules, (
        "importing train.reward_model pulled torch into the test process. Move "
        "the offending import inside the function that needs it."
    )
