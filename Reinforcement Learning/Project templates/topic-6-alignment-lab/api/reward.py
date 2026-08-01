"""
api/reward.py — text in, one number out, in NumPy alone.

This is the file the architecture note is about. Read the imports: NumPy, plus
one regex helper from `shared.preprocess`. No transformer, no tokeniser
library, no scikit-learn, no PyTorch. The deployed reward model is

    tokens = tokenise(text)                # shared/preprocess.py
    x      = tfidf(tokens, vocab, idf)     # this file, ~15 lines
    h      = relu(W0 @ x + b0)
    r      = (W1 @ h + b1)[0]              # a scalar

and that is the whole of it. A student who has believed for five topics that
"inference is matrix arithmetic and nothing more" gets to see the claim survive
contact with language.

WHY THE TF-IDF HEAD IS THE ONE THAT SHIPS
-----------------------------------------
The embedding head trained in `train/reward_model.py` is stronger. It is also
undeployable under this course's constraints, because computing its input
requires running a transformer — 250 MB of weights and PyTorch underneath them,
against a 690 MB guarantee. The TF-IDF head's input is computable from a
vocabulary array and an IDF vector that together weigh a few hundred kilobytes
and travel inside the same `.npz` as the weights.

That is a real deployment trade, not a course-specific dodge: you ship the
model whose FEATURE PIPELINE fits the budget, not the model with the best
held-out number. Report both accuracies, ship the cheaper one, and say in the
model card what the cheaper one cost you.

THE REPLICATION CONTRACT
------------------------
`tfidf_vector` below reproduces `sklearn.feature_extraction.text.TfidfVectorizer`
with its DEFAULT settings, which are the settings `train/reward_model.py` uses:

    lowercase=True, token_pattern=shared.preprocess.TOKEN_PATTERN,
    sublinear_tf=False, smooth_idf=True, norm="l2", binary=False

Under those settings, for term t in document d:

    tf(t, d)  = raw count of t in d
    idf(t)    = ln((1 + n_documents) / (1 + df(t))) + 1        [smooth_idf]
    v_t       = tf(t, d) * idf(t)
    x         = v / ||v||_2                                    [norm="l2"]

The IDF vector is fitted on the TRAINING split and exported into the archive.
Recomputing IDF from live traffic would be the text-domain version of
recomputing observation-normalisation statistics at serving time — the exact
train/serve skew `shared/preprocess.py` exists to prevent.

If `tests/test_equivalence.py` fails, this replication is the first place to
look, ahead of the weights. The usual causes, in order:

    1. A different tokeniser. Both sides must call `shared.preprocess.tokenise`
       (the serving side) or hand it `TOKEN_PATTERN` (the training side).
    2. `smooth_idf` or `norm` changed on the training side and not here.
    3. A transposed weight matrix — see api/policy.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.preprocess import tokenise


def relu(x: np.ndarray) -> np.ndarray:
    """max(0, x), elementwise.

    Written as `np.maximum(0.0, x)` rather than `x * (x > 0)` because the
    second form silently promotes an integer input to a boolean product and
    returns a different dtype than it was given.
    """
    return np.maximum(0.0, x)


def sigmoid(x: float | np.ndarray) -> np.ndarray:
    """Numerically stable logistic function.

    The naive `1 / (1 + exp(-x))` overflows to `inf` for x around -750 and
    returns 0.0 where the correct answer is a denormal — harmless — but raises
    a RuntimeWarning that turns into a log-flooding incident under load. The
    branch below never exponentiates a positive number.

    This matters here specifically because `POST /compare` returns
    `sigmoid(margin)` and the margin is UNBOUNDED: a Bradley-Terry reward has
    no scale, so an adversarial input can drive it wherever it likes.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


# ---------------------------------------------------------------------------
# The featuriser
# ---------------------------------------------------------------------------


def tfidf_vector(
    text: str,
    vocab_index: dict[str, int],
    idf: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Return (l2-normalised tf-idf vector, token count, out-of-vocabulary rate).

    The OOV rate is returned rather than discarded because it is the one honest
    signal this head can give about whether it is out of its depth. A response
    written in a register the training corpus never contained produces a nearly
    empty vector, and the head will then return roughly `b1` — a confident,
    stable, entirely uninformative number. `POST /score` surfaces this so the
    caller can distinguish "scored low" from "not really scored".

    Note the zero-vector guard. `v / ||v||` with an all-zero `v` is 0/0: NumPy
    returns NaN with a warning, the NaN propagates through both matrix
    multiplies, and FastAPI then fails to serialise a NaN float into JSON and
    returns a 500. The failing input is a one-word message, which real users
    send constantly.
    """
    tokens = tokenise(text)
    v = np.zeros(idf.shape[0], dtype=np.float64)
    in_vocab = 0
    for tok in tokens:
        j = vocab_index.get(tok)
        if j is not None:
            v[j] += 1.0          # raw term frequency; sublinear_tf=False
            in_vocab += 1
    v *= idf                     # tf * idf
    norm = float(np.linalg.norm(v))
    if norm > 0.0:
        v /= norm                # norm="l2"
    n_tokens = len(tokens)
    oov = 0.0 if n_tokens == 0 else 1.0 - (in_vocab / n_tokens)
    return v, n_tokens, oov


# ---------------------------------------------------------------------------
# The head
# ---------------------------------------------------------------------------


@dataclass
class RewardHead:
    """A scorer loaded from an `.npz`. Not a controller — it has no actions.

    Expected archive keys, written by `train/export.py::export_reward_head`:

        vocab  : (V,) unicode array, the fitted vocabulary IN COLUMN ORDER
        idf    : (V,) float, the fitted inverse document frequencies
        W0, b0 : (H, V), (H,)     first Linear layer
        W1, b1 : (1, H), (1,)     output layer, a single scalar

    `vocab` is a NumPy unicode array specifically so that the archive stays
    loadable with `allow_pickle=False`. A Python dict or a list of strings
    would require pickling, and a pickle is arbitrary code execution wearing a
    model's name — the serving tier must never load one.

    The vocabulary-to-index dict is built ONCE at load time. Rebuilding it per
    request turns an O(tokens) scoring call into an O(vocabulary) one, which at
    V = 2,000 is the difference between 40 microseconds and 4 milliseconds and
    is entirely invisible until the endpoint is under load.
    """

    vocab: np.ndarray
    idf: np.ndarray
    layers: list[tuple[np.ndarray, np.ndarray]]

    kind = "reward-head"

    def __post_init__(self) -> None:
        if self.vocab.shape[0] != self.idf.shape[0]:
            raise ValueError(
                f"vocab has {self.vocab.shape[0]} entries but idf has "
                f"{self.idf.shape[0]} — the archive is inconsistent and every "
                "score it produced would be silently misaligned"
            )
        if self.layers[0][0].shape[1] != self.vocab.shape[0]:
            raise ValueError(
                f"W0 expects {self.layers[0][0].shape[1]} features but the "
                f"vocabulary has {self.vocab.shape[0]}"
            )
        if self.layers[-1][0].shape[0] != 1:
            raise ValueError(
                "a reward head must output exactly one number; this archive's "
                f"final layer outputs {self.layers[-1][0].shape[0]}. Did you "
                "export a policy where a reward model was expected?"
            )
        self._index = {str(w): i for i, w in enumerate(self.vocab)}

    @property
    def obs_dim(self) -> int:
        return int(self.vocab.shape[0])

    @property
    def n_actions(self) -> int | None:
        # Deliberately None. A reward model does not choose anything, and
        # returning 1 here would let it be registered as a one-action policy
        # and served through /act, where it would return action 0 forever.
        return None

    def features(self, text: str) -> tuple[np.ndarray, int, float]:
        return tfidf_vector(text, self._index, self.idf)

    def score(self, text: str) -> tuple[float, int, float]:
        """Return (reward, token count, oov rate)."""
        x, n_tokens, oov = self.features(text)
        h = x
        for i, (W, b) in enumerate(self.layers):
            h = W @ h + b
            if i < len(self.layers) - 1:      # no activation on the output layer
                h = relu(h)
        return float(h[0]), n_tokens, oov

    def act(self, state: np.ndarray, deterministic: bool = True):
        """Refuse, loudly.

        `PolicyArtifactStore` returns whatever is registered, and `/act` calls
        `.act()` on it. Without this method a reward head reached through /act
        would raise AttributeError and surface as a 500; with it, the caller
        gets a 422 naming the endpoint they should have used. An error contract
        is part of the product.
        """
        raise ValueError(
            "this artifact is a reward head, not a control policy: it scores "
            "text and has no actions. Use POST /score or POST /compare."
        )
