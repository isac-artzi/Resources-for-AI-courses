"""Synthetic *dialect* character corpus (distinct Markov transition grammars).

Lives in ``shared`` so BOTH the API tier (to train) and the db/seed script import
the same generating functions. Instead of downloading a real text corpus (a
network + size cost that is unfriendly to free-tier builds), we synthesize
character strings from a tiny, KNOWN family of grammars:

    A string of class ``k`` is a walk over a first-order Markov chain whose
    transition matrix is ``STAY * shift_by(SHIFTS[k]) + (1 - STAY) * uniform``.
    Each chain prefers to advance the current character by a class-specific
    cyclic offset (dialect-1 nudges a->b->c->d->a, dialect-2 skips two, ...).

The transition matrices are **doubly stochastic**, so every class has the SAME
uniform stationary (unigram) distribution over characters. The classes differ
ONLY in their *transition* structure. This is the whole point: a model must
learn character-to-character dependencies -- a bag-of-characters view is useless
-- so a randomly-initialised network sits near chance while a network that
learned the language during pretraining transfers cleanly.

Two tasks share the alphabet:
  * **pretraining** (self-supervised): next-character language modelling over a
    corpus drawn from all dialects -- the model learns each dialect's transitions,
  * **fine-tuning** (supervised): classify a string by its dialect.

Only dataset *parameters* (n_rows, seq_len, seed, ...) are persisted to Supabase;
the strings are regenerated deterministically from the seed at train time.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# Vocabulary layout:
#   0                  -> PAD (reserved, never a real character)
#   1 .. len(ALPHABET) -> the character alphabet
PAD = 0
ALPHABET = "abcd"
CHAR_TO_ID = {c: i + 1 for i, c in enumerate(ALPHABET)}
ID_TO_CHAR = {i + 1: c for i, c in enumerate(ALPHABET)}
VOCAB_SIZE = len(ALPHABET) + 1                 # ids 0 .. 4
SEQ_LEN = 20

SHIFTS = [1, 2, 3]                             # class-specific cyclic offsets
STAY = 0.7                                     # weight on the preferred transition
CLASS_NAMES: List[str] = [f"dialect-{s}" for s in SHIFTS]
N_CLASSES = len(CLASS_NAMES)


def encode(text: str) -> List[int]:
    """Map a string over ALPHABET to token ids (unknown chars -> PAD)."""
    return [CHAR_TO_ID.get(c, PAD) for c in text]


def decode(ids: List[int]) -> str:
    """Map token ids back to a string (PAD -> '.')."""
    return "".join(ID_TO_CHAR.get(int(i), ".") for i in ids)


def _transition_matrices() -> List[np.ndarray]:
    """One doubly-stochastic transition matrix per dialect.

    ``STAY * P_shift + (1 - STAY) * uniform`` where ``P_shift`` is a cyclic
    permutation. Both terms are doubly stochastic, so the stationary (unigram)
    distribution is uniform for every class; only the transitions differ.
    """
    a = len(ALPHABET)
    uniform = np.ones((a, a)) / a
    mats = []
    for shift in SHIFTS:
        perm = np.zeros((a, a))
        for i in range(a):
            perm[i, (i + shift) % a] = 1.0
        mats.append(STAY * perm + (1.0 - STAY) * uniform)
    return mats


_MATS = _transition_matrices()


def _walk(rng: np.random.Generator, seq_len: int, k: int) -> np.ndarray:
    """Sample one length-``seq_len`` walk over dialect ``k``'s Markov chain."""
    mat = _MATS[k]
    seq = np.empty(seq_len, dtype=np.int64)
    cur = int(rng.integers(0, len(ALPHABET)))
    for t in range(seq_len):
        seq[t] = cur + 1                       # +1: id 0 is PAD
        cur = int(rng.choice(len(ALPHABET), p=mat[cur]))
    return seq


def generate_corpus(n_rows: int, seq_len: int = SEQ_LEN, seed: int = 42) -> np.ndarray:
    """Unlabelled strings from all dialects, for next-char LM pretraining."""
    rng = np.random.default_rng(seed)
    seqs = np.empty((n_rows, seq_len), dtype=np.int64)
    for i in range(n_rows):
        seqs[i] = _walk(rng, seq_len, int(rng.integers(0, N_CLASSES)))
    return seqs


def generate_labeled(
    n_rows: int, seq_len: int = SEQ_LEN, seed: int = 7
) -> Tuple[np.ndarray, List[int]]:
    """(strings, class labels) for the downstream dialect-classification task."""
    rng = np.random.default_rng(seed)
    seqs = np.empty((n_rows, seq_len), dtype=np.int64)
    labels: List[int] = []
    for i in range(n_rows):
        label = int(rng.integers(0, N_CLASSES))
        seqs[i] = _walk(rng, seq_len, label)
        labels.append(label)
    return seqs, labels


def generate_one(
    seq_len: int = SEQ_LEN, seed: int = 0, kind: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """Return ``(sequence [seq_len], label)`` for one classification sample."""
    rng = np.random.default_rng(seed)
    label = int(rng.integers(0, N_CLASSES)) if kind is None else int(kind)
    seq = _walk(rng, seq_len, label)
    return seq, label
