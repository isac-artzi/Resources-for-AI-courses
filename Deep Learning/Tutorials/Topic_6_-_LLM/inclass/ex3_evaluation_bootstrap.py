"""
AIT-204 · Topic 6 · In-class Exercise 3 — Is it really better? F1, ROUGE-L, perplexity and the bootstrap
Time: ~15–20 min (+ optional Part D at home)   |   Needs: numpy only

    classification (variant A)  : precision, recall, F1
    summarization  (variant B)  : ROUGE-L, built on the longest common subsequence
    language models             : perplexity = exp(average negative log-likelihood)
    any metric                  : a 95% bootstrap confidence interval, and one for Δ vs. zero-shot

Fill in every "TODO". The asserts check your code against the Part B hand calculations.
"""
import numpy as np


# ---------------------------------------------------------------------------
# TODO 1 — precision, recall, F1 for the positive class (1 = spam)
# ---------------------------------------------------------------------------
def prf1(y_true, y_pred):
    """y_true, y_pred: int arrays of 0/1. Returns (precision, recall, f1). Use 0.0 when a denominator is 0."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    # TODO 1: count fp and fn the same way, then compute precision, recall and f1.
    precision = recall = f1 = None  # ← your code here
    return precision, recall, f1


def f1_score(y_true, y_pred):
    return prf1(y_true, y_pred)[2]


# ---------------------------------------------------------------------------
# TODO 2 — longest common subsequence, the engine inside ROUGE-L
# ---------------------------------------------------------------------------
def lcs_len(a, b):
    """a, b: lists of tokens. Length of the longest sequence that appears in both, in order,
    not necessarily next to each other. Classic dynamic programming:
        L[i][j] = L[i-1][j-1] + 1                  if a[i-1] == b[j-1]
                = max(L[i-1][j], L[i][j-1])        otherwise"""
    L = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            # TODO 2: fill in L[i][j] using the recurrence above
            pass  # ← your code here
    return L[len(a)][len(b)]


def rouge_l(candidate, reference):
    """F1 form of ROUGE-L on whitespace tokens. Returns (precision, recall, f1)."""
    c, r = candidate.lower().split(), reference.lower().split()
    lcs = lcs_len(c, r)
    p, rec = lcs / len(c), lcs / len(r)
    return p, rec, (2 * p * rec / (p + rec) if p + rec else 0.0)


# ---------------------------------------------------------------------------
# TODO 3 — perplexity
# ---------------------------------------------------------------------------
def perplexity(p_correct):
    """p_correct: the probability the model gave to each TRUE next token. PPL = exp(−mean(ln p))."""
    # TODO 3: one line
    return None  # ← your code here


# ---------------------------------------------------------------------------
# TODO 4 — the bootstrap: a confidence interval for one model, and for the difference of two
# ---------------------------------------------------------------------------
def bootstrap_ci(y_true, y_pred, metric=f1_score, n_boot=2000, seed=0):
    """Resample the test set WITH replacement n_boot times, recompute the metric each time,
    return (2.5th percentile, 97.5th percentile)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng, n, scores = np.random.default_rng(seed), len(y_true), []
    for _ in range(n_boot):
        # TODO 4a: draw n indices with replacement (rng.integers), score that resample, append to scores
        pass  # ← your code here
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def bootstrap_delta_ci(y_true, pred_a, pred_b, metric=f1_score, n_boot=2000, seed=0):
    """PAIRED bootstrap for Δ = metric(A) − metric(B). Both models are scored on the SAME resampled
    messages, so 'this resample happened to be easy' cancels out of the difference."""
    y_true, pred_a, pred_b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    rng, n, deltas = np.random.default_rng(seed), len(y_true), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # TODO 4b: append metric(A) − metric(B), both computed on the SAME idx
        pass  # ← your code here
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


# ---- Given: a small held-out test set and two models' predictions -----------
def make_test_set(seed=6):
    """200 SMS labels (1 = spam) plus predictions from a fine-tuned model and a zero-shot baseline.
    The zero-shot model makes every mistake the fine-tuned model makes, and more."""
    rng = np.random.default_rng(seed)
    y = (rng.random(200) < 0.5).astype(int)
    hard = rng.random(200)                       # one difficulty score per message
    pred_ft = np.where(hard < 0.11, 1 - y, y)    # fine-tuned: wrong on the hardest ~11%
    pred_zs = np.where(hard < 0.19, 1 - y, y)    # zero-shot:  wrong on the hardest ~19%
    return y, pred_ft, pred_zs


if __name__ == "__main__":
    # ---- TODO 1: the lecture-notes confusion matrix  TP=84  FN=16  FP=10  TN=90 ----
    y_true = np.array([1] * 100 + [0] * 100)
    y_pred = np.array([1] * 84 + [0] * 16 + [1] * 10 + [0] * 90)
    p, r, f = prf1(y_true, y_pred)
    print(f"precision {p:.3f}   recall {r:.3f}   F1 {f:.3f}")
    assert np.allclose([p, r, f], [0.894, 0.840, 0.866], atol=1e-3)
    print("✓ TODO 1: matches Part B and the lecture notes")

    # ---- TODO 2: ROUGE-L -----------------------------------------------------
    cand = "team shipped the model on friday"
    ref = "the team deployed the fine-tuned model friday"
    assert lcs_len(cand.split(), ref.split()) == 4
    p, r, f = rouge_l(cand, ref)
    print(f"ROUGE-L  precision {p:.3f}   recall {r:.3f}   F1 {f:.3f}")
    assert np.allclose([p, r, f], [4 / 6, 4 / 7, 0.615], atol=1e-3)
    print("✓ TODO 2: LCS = 4 → ROUGE-L F1 = 0.615")

    # ---- TODO 3: perplexity --------------------------------------------------
    print(f"perplexity of [0.5, 0.25, 0.125, 0.5] = {perplexity([0.5, 0.25, 0.125, 0.5]):.3f}")
    assert abs(perplexity([0.5, 0.25, 0.125, 0.5]) - 2 ** 1.75) < 1e-6
    assert abs(perplexity([1 / 30000] * 50) - 30000) < 1e-3      # random guessing on a 30K vocabulary
    print("✓ TODO 3: 2^1.75 = 3.364, and a uniform guess over V tokens has perplexity V")

    # ---- TODO 4: is fine-tuned really better than zero-shot? -------------------
    y, pred_ft, pred_zs = make_test_set()
    f_ft, f_zs = f1_score(y, pred_ft), f1_score(y, pred_zs)
    ci_ft, ci_zs = bootstrap_ci(y, pred_ft), bootstrap_ci(y, pred_zs)
    ci_d = bootstrap_delta_ci(y, pred_ft, pred_zs)
    print(f"fine-tuned  F1 {f_ft:.3f}   95% CI [{ci_ft[0]:.3f}, {ci_ft[1]:.3f}]")
    print(f"zero-shot   F1 {f_zs:.3f}   95% CI [{ci_zs[0]:.3f}, {ci_zs[1]:.3f}]")
    print(f"Δ (paired)     {f_ft - f_zs:+.3f}   95% CI [{ci_d[0]:+.3f}, {ci_d[1]:+.3f}]")
    assert ci_ft[0] < f_ft < ci_ft[1] and ci_zs[0] < f_zs < ci_zs[1]
    assert ci_zs[1] > ci_ft[0], "the two separate intervals overlap on this test set"
    assert ci_d[0] > 0, "…but the paired interval for Δ excludes 0"
    print("✓ TODO 4: the separate intervals overlap, yet the paired Δ interval excludes 0")

    # ---- Part D (bridge to FineTuneFleet): the rows your leaderboard needs -----
    rows = [
        dict(base_model="zero_shot", headline_metric=round(f_zs, 3), ci_low=round(ci_zs[0], 3), ci_high=round(ci_zs[1], 3)),
        dict(base_model="distilbert-base-uncased", headline_metric=round(f_ft, 3), ci_low=round(ci_ft[0], 3), ci_high=round(ci_ft[1], 3)),
    ]
    for row in rows:
        print("experiments row →", row)
    # QUESTION (handout): the README needs one sentence that reports Δ with its interval.
    # Write it. Then say what you would report if the Δ interval had been [−0.01, +0.09].
