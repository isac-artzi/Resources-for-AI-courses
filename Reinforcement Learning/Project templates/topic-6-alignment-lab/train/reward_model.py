"""
train/reward_model.py — TWO REWARD HEADS, ONE LOSS.

TRAINING TIER (imports torch and scikit-learn). Never imported by api/ or ui/.

    python -m train.reward_model --offline --pairs 2400

Both heads are fitted with the SAME pairwise Bradley-Terry loss on the SAME
comparisons. They differ in exactly one thing: what a response looks like on
the way in.

    tfidf      response text -> TfidfVectorizer -> R^V   -> 64 -> 1
    embedding  response text -> frozen encoder  -> R^D   -> 64 -> 1

That is the entire experimental design, and it is a required result of this
product rather than an aside. The question it answers is not "which model is
better" — it is **what does the cheap feature representation cost you**, which
is the question you will actually be asked when someone looks at the inference
bill.

THE BRADLEY-TERRY LOSS, DERIVED IN FOUR LINES
---------------------------------------------
The Bradley-Terry model says the probability a labeller prefers y_c to y_r is

    P(y_c > y_r | x) = sigma( r(x, y_c) - r(x, y_r) )

Taking the negative log-likelihood of the observed comparisons gives

    L = - E[ log sigma( r(x, y_c) - r(x, y_r) ) ]

and that is the whole objective. Three properties follow immediately and all
three matter downstream:

  * **Only DIFFERENCES are identified.** Add a constant c to r(x, ·) for a
    fixed prompt and the loss is unchanged. So the absolute score returned by
    `POST /score` is meaningless in isolation, which is why `ScoreResponse`
    says so and why `POST /compare` exists.
  * **The loss is unbounded below.** Nothing stops the model driving the margin
    to +inf on the training pairs; the only thing preventing it is capacity and
    early stopping. This is the first appearance in this course of an objective
    that rewards its own overconfidence, and it is the same structural fact
    that makes DPO at small beta degenerate.
  * **The gradient vanishes on easy pairs.** sigma'(m) -> 0 as m grows, so once
    a comparison is confidently right it stops teaching. That is a feature, and
    it is also why held-out accuracy plateaus long before the loss does.

WHAT THE COMPARISON MUST REPORT (build step 4)
----------------------------------------------
  (a) held-out pairwise accuracy for each head against the 50% baseline, and a
      plot of the reward-margin distribution for each;
  (b) a length-bias test for each — regress assigned reward on response length
      and report the correlation — naming which head is more susceptible and
      why that is what you would expect;
  (c) which head you would ship and what the cheaper one costs you.

`main()` computes (a) and (b) and writes them to `reports/reward_heads.json`
and `reports/*.png`. (c) is yours to argue in the README, and the argument is
not "the accurate one": the accurate one cannot be served under this course's
memory budget, which is the whole point of the exercise.

A NOTE ON THE 50% BASELINE
--------------------------
Every held-out pair has its chosen response in the `chosen` column, so a
CLASSIFIER given both responses at once could score 100% by answering "the
first one" and never reading the text. That is not possible here, and the
reason is structural rather than a precaution: the head is a function of ONE
response. It computes r(y) for the chosen and r(y) for the rejected in two
independent forward passes that share no state and cannot see each other, and
the comparison happens afterwards in `pairwise_accuracy`. There is no position
for a positional shortcut to attach to, so an uninformative head scores exactly
0.50 and the baseline is honest.

Keep that property if you extend this. A head that took a PAIR as input — a
cross-encoder, say — would need the pairs shuffled and both orders evaluated,
and without it you would report a stunning accuracy for a model that learned
column order.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass

import numpy as np

from shared.preprocess import TOKEN_PATTERN, response_length

REPORTS = pathlib.Path("reports")

# The two thresholds `tests/test_length_bias.py` enforces on the DEPLOYED head.
# Justified at length in that file; repeated here because this is where the
# numbers are produced and a threshold you cannot find is a threshold nobody
# revisits.
LENGTH_BIAS_MAX_R = 0.45
LENGTH_MATCHED_MIN_ACCURACY = 0.60


@dataclass
class HeadConfig:
    """Hyperparameters shared by both heads, so that only the FEATURES differ.

    If you change one of these for one head, the comparison stops being about
    representations and starts being about tuning, and the result in your
    README stops answering the question it claims to answer.
    """

    hidden: int = 64
    # 25, and it is a chosen number. Both heads' held-out accuracy peaks
    # between epoch 10 and 20 and then declines; by epoch 60 the TF-IDF head
    # has fallen from 0.88 to 0.81 while the embedding head has barely moved.
    # That asymmetry is a result worth reporting rather than a value to tune
    # away: THE CHEAPER REPRESENTATION IS ALSO THE ONE THAT OVERFITS SOONER,
    # because a bag of 74 term weights can memorise a 1,920-pair training set
    # and a 256-dimensional dense vector cannot memorise it as cheaply.
    #
    # There is deliberately NO early stopping. Stopping each head at its own
    # best epoch would make the comparison about two different training
    # procedures, and the honest version — one budget, both heads, the curve
    # printed in `history` so a reader can see where each one turned — is more
    # informative than a pair of tuned numbers.
    epochs: int = 25
    batch_size: int = 128
    lr: float = 3e-3
    weight_decay: float = 1e-4
    seed: int = 0
    # L2 on the reward scale. The Bradley-Terry loss is unbounded below, so
    # without a mild pull toward zero the rewards drift to +-20 on a dataset
    # this size and `sigmoid(margin)` saturates to exactly 1.0, at which point
    # `POST /compare` reports certainty it does not have.
    reward_l2: float = 1e-3


# ===========================================================================
# Features
# ===========================================================================


def fit_tfidf(train_texts: list[str], max_features: int = 2000):
    """Fit the vectoriser on the TRAINING SPLIT ONLY, and return (vectoriser, vocab, idf).

    Fitting on train+test is the classic leak, and in a bag-of-words model it
    is not a small one: the IDF of a term is a statistic of the corpus, so a
    vectoriser fitted on the test responses has already been told which words
    are rare in the data it will be evaluated on.

    Every keyword below is passed EXPLICITLY even where it is the default. The
    serving-side replication in `api/reward.py` is written against these exact
    settings, so leaving them implicit means a scikit-learn release can change
    the deployed model's arithmetic without changing a line of this repository.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        lowercase=True,
        token_pattern=TOKEN_PATTERN,   # shared with shared.preprocess.tokenise
        max_features=max_features,     # bounds the artifact: V floats of IDF + V strings
        sublinear_tf=False,            # replicated in api/reward.py
        smooth_idf=True,               # idf = ln((1+n)/(1+df)) + 1
        norm="l2",
        binary=False,
        # min_df=1 on purpose. A higher threshold would be defensible on real
        # text; here it would silently drop the rarest GOOD/BAD tokens, which
        # are precisely the ones carrying the signal.
        min_df=1,
    )
    vec.fit(train_texts)
    vocab = [t for t, _ in sorted(vec.vocabulary_.items(), key=lambda kv: kv[1])]
    return vec, vocab, np.asarray(vec.idf_, dtype=np.float64)


def tfidf_matrix(vec, texts: list[str]) -> np.ndarray:
    """Dense float32. Sparse would be better; dense is honest about the cost.

    At V = 2,000 and 4,800 responses this is 38 MB, which fits comfortably and
    keeps the training loop a dozen readable lines instead of a sparse-tensor
    exercise. At the real UltraFeedback scale it does not fit, and the fix is
    `torch.sparse` or a hashing vectoriser — say which you would use in your
    README rather than discovering it at 60,000 rows.
    """
    return np.asarray(vec.transform(texts).todense(), dtype=np.float32)


# ===========================================================================
# The head, and the one loss
# ===========================================================================


def build_head(in_dim: int, cfg: HeadConfig):
    """in_dim -> hidden -> 1. A ReLU MLP with a scalar output.

    Small on purpose. A reward model with more capacity than the comparison
    data supports will fit the length confound and the labeller noise, and the
    two things this product is trying to measure — held-out accuracy and length
    bias — both get worse in ways that look like bugs.
    """
    import torch

    torch.manual_seed(cfg.seed)
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, cfg.hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(cfg.hidden, 1),
    )


def bradley_terry_loss(reward_chosen, reward_rejected, l2: float = 0.0):
    """-log sigma(r_c - r_r), plus an optional pull toward zero reward.

    `logsigmoid` rather than `log(sigmoid(x))`: the second underflows to
    `log(0) = -inf` for margins below about -37 in float32, and one such pair
    in a batch turns the whole gradient into NaN. This is the same
    numerical-stability point as the max-subtraction in `softmax`, and it bites
    here for real — early in training, with random weights, margins of -40 do
    occur.
    """
    import torch

    margin = reward_chosen - reward_rejected
    loss = -torch.nn.functional.logsigmoid(margin).mean()
    if l2 > 0.0:
        # Penalise the reward SCALE, not the weights. Weight decay on a
        # two-layer net does not bound the output; this does, and bounding the
        # output is what keeps sigmoid(margin) away from exactly 1.0.
        loss = loss + l2 * (reward_chosen.pow(2).mean() + reward_rejected.pow(2).mean())
    return loss


def train_head(
    Xc_train: np.ndarray,
    Xr_train: np.ndarray,
    Xc_test: np.ndarray,
    Xr_test: np.ndarray,
    cfg: HeadConfig,
):
    """Fit one head. Returns (module, history).

    The loop is deliberately plain: shuffle, minibatch, forward both sides,
    Bradley-Terry loss, step. There is no scheduler and no early stopping,
    because the comparison this file exists to make requires the two heads to
    be trained identically and every mechanism added is a mechanism that could
    interact differently with the two feature spaces.
    """
    import torch

    torch.manual_seed(cfg.seed)
    model = build_head(Xc_train.shape[1], cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    tc = torch.from_numpy(np.ascontiguousarray(Xc_train))
    tr = torch.from_numpy(np.ascontiguousarray(Xr_train))
    n = tc.shape[0]
    g = torch.Generator().manual_seed(cfg.seed)

    history: list[dict] = []
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n, generator=g)
        epoch_loss = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            rc = model(tc[idx]).squeeze(-1)
            rr = model(tr[idx]).squeeze(-1)
            loss = bradley_terry_loss(rc, rr, cfg.reward_l2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            # `.detach()` before the float cast. Without it torch warns on every
            # minibatch that a grad-tracking tensor is being converted to a
            # scalar, and 60 epochs of that noise is how a real warning later in
            # the run goes unread.
            epoch_loss += float(loss.detach()) * len(idx)
        # Logged every 5 epochs, not just at the end. The held-out curve is
        # where you SEE the overfitting noted on HeadConfig.epochs; a history
        # with two points cannot show a turn.
        if epoch % 5 == 0 or epoch == cfg.epochs - 1:
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": epoch_loss / n,
                    "test_accuracy": pairwise_accuracy(
                        predict(model, Xc_test), predict(model, Xr_test)
                    ),
                }
            )
    return model, history


def predict(model, X: np.ndarray) -> np.ndarray:
    """Scores for a feature matrix, as a 1-D NumPy array."""
    import torch

    with torch.no_grad():
        return model(torch.from_numpy(np.ascontiguousarray(X))).squeeze(-1).numpy()


# ===========================================================================
# Evaluation
# ===========================================================================


def pairwise_accuracy(r_chosen: np.ndarray, r_rejected: np.ndarray) -> float:
    """Fraction of held-out pairs the head ranks correctly. Chance is 0.50.

    Ties count as HALF, not as correct. A head that collapsed to a constant
    would otherwise score 100% under `>=` and 0% under `>`, and neither number
    describes it. Half is the accuracy a coin flip would give on a tie, which
    is what a tie is.
    """
    diff = np.asarray(r_chosen, dtype=np.float64) - np.asarray(r_rejected, dtype=np.float64)
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def length_bias(rewards: np.ndarray, lengths: np.ndarray) -> dict:
    """Regress assigned reward on response length. Report r, r^2 and the slope.

    This is build step 4(b) and the concrete form of the spurious-correlate
    problem. Three numbers rather than one, because they say different things:

      * `pearson_r` — the correlation, and the number the test thresholds on.
      * `r_squared` — the share of the head's own score variance that length
        alone explains. r = 0.35 sounds mild; r^2 = 0.12 says an eighth of what
        this model does is counting words.
      * `slope` — reward per additional token, in the head's own units. The one
        that tells you how much padding is worth to an optimiser pointed at it.

    The correlation is computed over ALL responses of the split, chosen and
    rejected pooled. Computing it on chosen responses alone would condition on
    the label and measure the data's length confound rather than the model's
    use of it.

    READ THIS NUMBER WITH THE DATA'S OWN CONFOUND IN FRONT OF YOU. In a corpus
    where longer responses really are better, a head that scores quality
    perfectly and ignores length entirely will STILL show a positive
    reward-length correlation, because quality and length are correlated in the
    data. `PreferenceDataset.summary()` reports `label_length_pearson_r` for
    exactly this comparison. A head at r = 0.32 against a corpus at r = 0.28 is
    not obviously doing anything wrong; a head at r = 0.7 against the same
    corpus is. This is why `length_decodability` and
    `length_matched_accuracy` below exist: the raw correlation is the number
    the build step asks for, and on its own it cannot distinguish a biased
    model from an honest one.
    """
    r = np.asarray(rewards, dtype=np.float64)
    x = np.asarray(lengths, dtype=np.float64)
    ok = np.isfinite(r) & np.isfinite(x)
    r, x = r[ok], x[ok]
    if r.size < 3 or x.std() == 0 or r.std() == 0:
        return {"pearson_r": float("nan"), "r_squared": float("nan"), "slope": float("nan"),
                "n": int(r.size)}
    pr = float(np.corrcoef(x, r)[0, 1])
    slope = float(np.polyfit(x, r, 1)[0])
    return {"pearson_r": pr, "r_squared": pr * pr, "slope": slope, "n": int(r.size)}


def paired_accuracy_difference(
    correct_a: np.ndarray, correct_b: np.ndarray
) -> dict:
    """Compare two heads on the SAME pairs, correctly. McNemar's test.

    The two heads are evaluated on identical held-out comparisons, so the
    unpaired standard error — `sqrt(se_a^2 + se_b^2)` — is the WRONG one and it
    is wrong in the direction that matters. On this template's offline corpus
    it gives +/- 0.021 for a difference of 0.027, which reads as "not
    significant"; the paired calculation gives +/- 0.009 and z = 3.0, which
    reads as "significant at p ~ 0.003". Same data, opposite conclusion, and
    the paired one is right: most of the variance in either head's accuracy is
    variance in WHICH PAIRS ARE HARD, and that is shared between them and
    cancels.

    The paired standard error of the accuracy difference is `sqrt(b + c) / n`,
    where b and c count the DISCORDANT pairs — the ones exactly one head got
    right. Concordant pairs carry no information about the difference at all,
    which is the whole insight.

    Report this rather than two independent error bars. "The stronger head is
    better by 0.027 +/- 0.021" invites the reader to conclude nothing; it is
    also not what the experiment measured.
    """
    a = np.asarray(correct_a, dtype=bool)
    b_only = int(np.sum(a & ~np.asarray(correct_b, dtype=bool)))
    c_only = int(np.sum(~a & np.asarray(correct_b, dtype=bool)))
    n = a.size
    discordant = b_only + c_only
    diff = (c_only - b_only) / n
    se = float(np.sqrt(discordant) / n) if discordant else float("nan")
    return {
        "accuracy_difference": float(diff),
        "paired_stderr": se,
        "z": float(diff / se) if discordant and se > 0 else float("nan"),
        "b_only_first_head_right": b_only,
        "c_only_second_head_right": c_only,
        "n_pairs": int(n),
        "n_discordant": discordant,
    }


def length_decodability(
    X_train: np.ndarray,
    len_train: np.ndarray,
    X_test: np.ndarray,
    len_test: np.ndarray,
) -> float:
    """Held-out R^2 of a LINEAR probe predicting response length from the features.

    This is the SUSCEPTIBILITY measurement, and it is the one that separates
    the two heads. `length_bias` asks what the fitted head happened to do;
    this asks what its feature space would let it do — how cheaply length is
    available to be used. A representation from which length is trivially
    decodable is a representation in which "prefer longer" is a low-complexity
    hypothesis, and a Bradley-Terry loss on a corpus where longer really is
    better will find low-complexity hypotheses first.

    Ridge rather than ordinary least squares because V can exceed n and the
    normal equations are then singular; alpha=1.0 is small enough not to
    matter at these dimensions and is not tuned, because a tuned probe would
    be measuring the tuning.
    """
    from sklearn.linear_model import Ridge

    probe = Ridge(alpha=1.0).fit(X_train, len_train)
    pred = probe.predict(X_test)
    y = np.asarray(len_test, dtype=np.float64)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def length_matched_accuracy(
    r_chosen: np.ndarray,
    r_rejected: np.ndarray,
    len_chosen: np.ndarray,
    len_rejected: np.ndarray,
    tolerance: int = 2,
) -> dict:
    """Held-out accuracy restricted to pairs whose two responses are nearly the same length.

    THE DIAGNOSTIC THAT ACTUALLY ANSWERS THE QUESTION, and the one to run on
    real data where you have no ground-truth quality to correlate against.
    Restricting to length-matched pairs removes the confound by conditioning
    rather than by modelling: within this subset, "prefer the longer one" is
    worth nothing, so whatever accuracy survives is accuracy the head earned
    from the text.

    A head whose full-set accuracy is 0.80 and whose length-matched accuracy is
    0.55 has been reading a ruler. One that holds up — 0.80 to 0.77 — has not.
    Report both; the DROP is the finding, not either number alone.

    `tolerance` is in tokens and is a real trade: 0 gives an unimpeachable
    subset and, on most corpora, almost no rows. 2 keeps enough pairs for the
    accuracy to have a usable standard error. Report `n_matched` alongside the
    accuracy so a reader can see which side of that trade you landed on.
    """
    lc = np.asarray(len_chosen, dtype=np.float64)
    lr = np.asarray(len_rejected, dtype=np.float64)
    mask = np.abs(lc - lr) <= tolerance
    n = int(mask.sum())
    if n < 20:
        # Refuse to report an accuracy from a handful of pairs. A number with a
        # standard error of 0.15 is not evidence, and printing it anyway is how
        # it ends up in a table without the error bar.
        return {"n_matched": n, "accuracy": float("nan"),
                "note": "too few length-matched pairs to report; widen --length-tolerance"}
    acc = pairwise_accuracy(np.asarray(r_chosen)[mask], np.asarray(r_rejected)[mask])
    return {
        "n_matched": n,
        "accuracy": acc,
        "stderr": float(np.sqrt(acc * (1 - acc) / n)),
        "tolerance_tokens": tolerance,
    }


def _plot_margins(margins: dict[str, np.ndarray], path: pathlib.Path) -> None:
    """Overlaid histograms of the held-out reward margin, one series per head.

    `matplotlib.use("Agg")` before pyplot is imported, and before anything
    else: on a headless machine — Colab, CI, a container — the default backend
    tries to reach a display server and either hangs or raises, and the
    traceback names Tk rather than the plot.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, m in margins.items():
        acc = float((m > 0).mean())
        ax.hist(m, bins=50, alpha=0.55, label=f"{name} (acc {acc:.3f})")
    # The line at zero is the decision boundary: mass to its left is a pair the
    # head ranked backwards. A margin histogram without it is decoration.
    ax.axvline(0.0, color="k", lw=1.2, ls="--", label="decision boundary")
    ax.set_xlabel("held-out reward margin  r(chosen) - r(rejected)")
    ax.set_ylabel("pairs")
    ax.set_title("Reward-margin distribution, held-out split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_length_bias(series: dict[str, tuple[np.ndarray, np.ndarray]],
                      path: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(series), figsize=(5.5 * len(series), 4), squeeze=False)
    for ax, (name, (lengths, rewards)) in zip(axes[0], series.items()):
        st = length_bias(rewards, lengths)
        ax.scatter(lengths, rewards, s=6, alpha=0.25)
        xs = np.linspace(float(np.min(lengths)), float(np.max(lengths)), 50)
        b, a = np.polyfit(lengths, rewards, 1)[0], np.polyfit(lengths, rewards, 1)[1]
        ax.plot(xs, b * xs + a, color="crimson", lw=2)
        ax.set_title(f"{name}: r = {st['pearson_r']:+.3f}  (r^2 = {st['r_squared']:.3f})")
        ax.set_xlabel("response length (tokens)")
        ax.set_ylabel("assigned reward")
    fig.suptitle("Length bias: assigned reward regressed on response length")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ===========================================================================
# The comparison
# ===========================================================================


def train_both(
    dataset,
    embeddings: dict[str, tuple[np.ndarray, np.ndarray]],
    cfg: HeadConfig,
    max_features: int = 2000,
    tfidf_out: str | pathlib.Path = "policies/reward_tfidf.npz",
    embed_out: str | pathlib.Path = "policies/reward_embedding.npz",
    experiment_id: str | None = None,
    reports: pathlib.Path = REPORTS,
) -> dict:
    """Fit both heads, evaluate, plot, export and register. Returns the result dict."""
    from train.export import export_embedding_head, export_reward_head, register

    train_rows, test_rows = dataset.train, dataset.test

    # -- head 1: TF-IDF over raw response text ------------------------------
    train_texts = [r["chosen"] for r in train_rows] + [r["rejected"] for r in train_rows]
    vec, vocab, idf = fit_tfidf(train_texts, max_features=max_features)

    Xc_tr = tfidf_matrix(vec, [r["chosen"] for r in train_rows])
    Xr_tr = tfidf_matrix(vec, [r["rejected"] for r in train_rows])
    Xc_te = tfidf_matrix(vec, [r["chosen"] for r in test_rows])
    Xr_te = tfidf_matrix(vec, [r["rejected"] for r in test_rows])

    tfidf_model, tfidf_hist = train_head(Xc_tr, Xr_tr, Xc_te, Xr_te, cfg)
    tfidf_rc, tfidf_rr = predict(tfidf_model, Xc_te), predict(tfidf_model, Xr_te)

    # -- head 2: the cached embeddings --------------------------------------
    Ec_tr, Er_tr = embeddings["train"]
    Ec_te, Er_te = embeddings["test"]
    embed_model, embed_hist = train_head(Ec_tr, Er_tr, Ec_te, Er_te, cfg)
    embed_rc, embed_rr = predict(embed_model, Ec_te), predict(embed_model, Er_te)

    # -- (a) accuracy and margins, (b) the three length-bias measurements ----
    results: dict[str, dict] = {}
    margins: dict[str, np.ndarray] = {}

    len_c_te = np.asarray([response_length(r["chosen"]) for r in test_rows], dtype=np.float64)
    len_r_te = np.asarray([response_length(r["rejected"]) for r in test_rows], dtype=np.float64)
    lengths = np.concatenate([len_c_te, len_r_te])
    len_tr = np.asarray(
        [response_length(r["chosen"]) for r in train_rows]
        + [response_length(r["rejected"]) for r in train_rows],
        dtype=np.float64,
    )
    length_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for name, (rc, rr), hist, dim, (Ftr, Fte) in (
        ("tfidf", (tfidf_rc, tfidf_rr), tfidf_hist, Xc_tr.shape[1],
         (np.vstack([Xc_tr, Xr_tr]), np.vstack([Xc_te, Xr_te]))),
        ("embedding", (embed_rc, embed_rr), embed_hist, Ec_tr.shape[1],
         (np.vstack([Ec_tr, Er_tr]), np.vstack([Ec_te, Er_te]))),
    ):
        pooled = np.concatenate([rc, rr])
        acc = pairwise_accuracy(rc, rr)
        matched = length_matched_accuracy(rc, rr, len_c_te, len_r_te)
        results[name] = {
            "feature_dim": int(dim),
            "held_out_accuracy": acc,
            "baseline_accuracy": 0.5,
            "n_test_pairs": len(test_rows),
            # The standard error of a proportion, so "0.87 vs 0.81" can be read
            # as a difference or as noise instead of being asserted to be one.
            "accuracy_stderr": float(np.sqrt(acc * (1 - acc) / max(len(test_rows), 1))),
            "mean_margin": float(np.mean(rc - rr)),
            "length_bias": length_bias(pooled, lengths),
            "length_decodability_r2": length_decodability(Ftr, len_tr, Fte, lengths),
            "length_matched": matched,
            # The number that means something: how much accuracy survives when
            # the length shortcut is taken away. Computed here rather than in
            # the README so it cannot drift from the numbers around it.
            "length_matched_drop": (
                None if not np.isfinite(matched.get("accuracy", float("nan")))
                else round(acc - matched["accuracy"], 4)
            ),
            "history": hist,
        }
        margins[name] = rc - rr
        length_series[name] = (lengths, pooled)

    _plot_margins(margins, reports / "reward_margins.png")
    _plot_length_bias(length_series, reports / "length_bias.png")

    # -- export and register both -------------------------------------------
    tfidf_row = export_reward_head(tfidf_model, vocab, idf, tfidf_out)
    register(tfidf_row, experiment_id=experiment_id)
    embed_row = export_embedding_head(embed_model, Ec_tr.shape[1], embed_out)
    register(embed_row, experiment_id=experiment_id)

    results["tfidf"]["artifact"] = tfidf_row
    results["embedding"]["artifact"] = embed_row
    # The comparison the product exists to make, tested PAIRED. See the
    # docstring of `paired_accuracy_difference` for why the two independent
    # error bars above are the wrong instrument for this question.
    results["tfidf_vs_embedding"] = paired_accuracy_difference(
        tfidf_rc > tfidf_rr, embed_rc > embed_rr
    )
    results["deployed"] = "tfidf"
    results["length_bias_threshold"] = LENGTH_BIAS_MAX_R
    results["length_matched_min_accuracy"] = LENGTH_MATCHED_MIN_ACCURACY
    results["dataset"] = dataset.summary()

    reports.mkdir(parents=True, exist_ok=True)
    (reports / "reward_heads.json").write_text(json.dumps(results, indent=2))
    return results


def dump_reference(out_dir: str | pathlib.Path, seed: int = 0, n_probe: int = 96,
                   epochs: int = 8, n_pairs: int = 600) -> dict:
    """Train a small head, export it, and record the PyTorch + scikit-learn scores.

    Written for `tests/test_equivalence.py`, which runs it in a subprocess. The
    test then loads the exported `.npz` with the NumPy `RewardHead` and compares
    the two score vectors.

    It trains a fresh head rather than loading `policies/reward_tfidf.npz`
    because the exported archive contains no scikit-learn vectoriser — that is
    the point of the export — so there would be no reference implementation
    left to disagree with. Reconstructing a `TfidfVectorizer` from the exported
    vocabulary and IDF would mean the test compared the NumPy path against a
    vectoriser that this repository built, rather than against scikit-learn's
    own transform, and the failure mode it exists to catch is precisely a
    divergence from scikit-learn.

    THE PROBE SET IS ADVERSARIAL ON PURPOSE. Held-out responses agree easily;
    what breaks a hand-written featuriser is an empty document (0/0 in the L2
    normalisation), a fully out-of-vocabulary document (the zero vector), a
    single repeated token (the L2 norm collapses onto one coordinate), mixed
    case and punctuation (the tokeniser), and a document much longer than
    anything in training.
    """
    import torch

    from train.data import load_synthetic
    from train.export import export_reward_head

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_synthetic(n_pairs=n_pairs, seed=seed)
    train_texts = [r["chosen"] for r in ds.train] + [r["rejected"] for r in ds.train]
    vec, vocab, idf = fit_tfidf(train_texts)
    cfg = HeadConfig(seed=seed, epochs=epochs)
    model, _ = train_head(
        tfidf_matrix(vec, [r["chosen"] for r in ds.train]),
        tfidf_matrix(vec, [r["rejected"] for r in ds.train]),
        tfidf_matrix(vec, [r["chosen"] for r in ds.test]),
        tfidf_matrix(vec, [r["rejected"] for r in ds.test]),
        cfg,
    )
    row = export_reward_head(model, vocab, idf, out / "equivalence_head.npz")

    probe = [r["chosen"] for r in ds.test[: max(n_probe - 8, 1)]]
    probe += [
        "",                                        # empty: the 0/0 case
        "   \n\t  ",                               # whitespace only
        "a I x",                                   # tokens too short for \b\w\w+\b
        "zzqqxx yyzzww unknowable wordsalad",      # fully out of vocabulary
        "verified " * 40,                          # one token, repeated: L2 collapse
        "SPECIFIC, Evidence; TESTED!  baseline.",  # case and punctuation
        " ".join(train_texts[0].split() * 6),      # much longer than training
        train_texts[1],                            # an in-distribution control
    ]

    with torch.no_grad():
        X = torch.from_numpy(tfidf_matrix(vec, probe))
        scores = model(X).squeeze(-1).numpy().astype(float).tolist()

    payload = {"texts": probe, "scores": scores, "artifact": str(out / "equivalence_head.npz"),
               "vocab_size": len(vocab), "artifact_row": row}
    (out / "reference.json").write_text(json.dumps(payload))
    return {"artifact": payload["artifact"], "n_probe": len(probe), "vocab_size": len(vocab)}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Train and compare the two reward heads.")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--pairs", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=HeadConfig.epochs)
    ap.add_argument("--max-features", type=int, default=2000)
    args = ap.parse_args(argv)

    from train.data import load_preferences, persist
    from train.embed import embed_dataset, make_encoder

    ds = load_preferences(args.offline, n_pairs=args.pairs, seed=args.seed)
    persist(ds)
    enc = make_encoder(args.offline)
    emb, _ = embed_dataset(ds, enc)

    cfg = HeadConfig(seed=args.seed, epochs=args.epochs)
    res = train_both(ds, emb, cfg, max_features=args.max_features)

    print(
        json.dumps(
            {
                "data_label_length_r": round(
                    res["dataset"]["test"]["label_length_pearson_r"], 4
                ),
                **{
                    k: {
                        "held_out_accuracy": round(v["held_out_accuracy"], 4),
                        "accuracy_stderr": round(v["accuracy_stderr"], 4),
                        "length_bias_r": round(v["length_bias"]["pearson_r"], 4),
                        "length_decodability_r2": round(v["length_decodability_r2"], 4),
                        "length_matched_accuracy": round(
                            v["length_matched"].get("accuracy", float("nan")), 4
                        ),
                        "length_matched_drop": v["length_matched_drop"],
                        "feature_dim": v["feature_dim"],
                        "artifact_kb": round(v["artifact"]["bytes"] / 1024, 1),
                    }
                    for k, v in res.items()
                    if isinstance(v, dict) and "held_out_accuracy" in v
                },
            },
            indent=2,
        )
    )
    return res


if __name__ == "__main__":  # pragma: no cover - a CLI entry point
    main()
