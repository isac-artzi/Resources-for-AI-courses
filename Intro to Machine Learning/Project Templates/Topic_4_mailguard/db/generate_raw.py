"""Generate a synthetic spam / ham (not-spam) email corpus.

A spam-filter template needs labelled text. Real spam datasets are large and
awkward to redistribute, so we generate a *deterministic* synthetic corpus
(fixed random seed). Each message is a bag of words drawn from three word lists:

    * spammy words  -- appear mostly in spam,
    * hammy words    -- appear mostly in legitimate mail,
    * neutral words  -- appear in both (so the classes are not trivially split).

The mild overlap keeps the problem realistic: the model must weigh the evidence
rather than memorize a single give-away word.

The output is committed to the repo, so you normally do NOT need to run this.
To use a real corpus instead, replace ``generate`` with a loader that returns
the same two columns (``label`` in {"spam","ham"}, ``text``).

    python db/generate_raw.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import RAW_DIR  # noqa: E402  (import after sys.path fix)

N_PER_CLASS = 400  # 400 spam + 400 ham = 800 messages
SEED = 42

# Words that lean spammy, hammy, or neutral. Kept small and obvious so the
# behaviour is easy to reason about in class.
SPAM_WORDS = [
    "free", "win", "winner", "cash", "prize", "offer", "click", "urgent",
    "guaranteed", "credit", "loan", "cheap", "discount", "buy", "now", "limited",
]
HAM_WORDS = [
    "meeting", "project", "report", "schedule", "team", "lunch", "review",
    "deadline", "attached", "budget", "notes", "agenda", "call", "update", "draft",
]
NEUTRAL_WORDS = ["the", "please", "you", "your", "and", "for", "this", "we", "to", "a"]


def _make_message(rng, primary, primary_words, other_words):
    """Build one message: mostly ``primary_words`` + neutral, a little leakage."""
    n_primary = rng.integers(4, 9)      # 4-8 topic words
    n_neutral = rng.integers(2, 5)      # 2-4 filler words
    n_leak = rng.integers(0, 2)         # 0-1 word from the OTHER class

    words = list(rng.choice(primary_words, size=n_primary))
    words += list(rng.choice(NEUTRAL_WORDS, size=n_neutral))
    words += list(rng.choice(other_words, size=n_leak))
    rng.shuffle(words)
    return " ".join(words)


def generate(n_per_class: int = N_PER_CLASS, seed: int = SEED) -> pd.DataFrame:
    """Build a balanced, shuffled spam/ham DataFrame."""
    rng = np.random.default_rng(seed)

    rows = []
    for _ in range(n_per_class):
        rows.append(("spam", _make_message(rng, "spam", SPAM_WORDS, HAM_WORDS)))
    for _ in range(n_per_class):
        rows.append(("ham", _make_message(rng, "ham", HAM_WORDS, SPAM_WORDS)))

    df = pd.DataFrame(rows, columns=["label", "text"])
    # Shuffle so spam and ham are interleaved (not all spam then all ham).
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "emails.csv"
    generate().to_csv(out, index=False)
    print(f"Wrote {out}")
