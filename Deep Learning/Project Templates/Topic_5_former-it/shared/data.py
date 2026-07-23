"""Synthetic *palindrome* sequence generator.

Lives in ``shared`` so BOTH the API tier (to train) and the db/seed script import
the same generating function. Instead of downloading a real corpus (a network +
size cost that is unfriendly to free-tier builds), we synthesize short
integer-symbol sequences and pose a simple, KNOWN algorithmic task:

    Is the sequence a **palindrome** (reads the same forwards and backwards)?

Half the sequences are palindromes (draw the first half, mirror it); the other
half are non-palindromes (drawn at random, rejecting any that happen to be
palindromic). Because the label depends on comparing position ``i`` with
position ``n-1-i``, a self-attention head can solve it by attending along the
**anti-diagonal** -- which is exactly what the per-head attention heatmaps in the
UI let a viewer see.

Only dataset *parameters* (n_rows, seq_len, seed, ...) are persisted to Supabase;
the sequences are regenerated deterministically from the seed at train time.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# Vocabulary layout:
#   0            -> PAD (reserved, never a real symbol)
#   1 .. N_SYMBOLS -> the symbol alphabet
PAD = 0
N_SYMBOLS = 5
VOCAB_SIZE = N_SYMBOLS + 1                      # ids 0 .. 5
SEQ_LEN = 8                                     # even, so every symbol has a mirror
CLASS_NAMES: List[str] = ["random", "palindrome"]
N_CLASSES = len(CLASS_NAMES)


def _random_symbols(rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.integers(1, VOCAB_SIZE, size=size, dtype=np.int64)


def _make_palindrome(rng: np.random.Generator, seq_len: int) -> np.ndarray:
    half = _random_symbols(rng, seq_len // 2)
    if seq_len % 2 == 0:
        return np.concatenate([half, half[::-1]])
    mid = _random_symbols(rng, 1)
    return np.concatenate([half, mid, half[::-1]])


def _make_non_palindrome(rng: np.random.Generator, seq_len: int) -> np.ndarray:
    while True:
        seq = _random_symbols(rng, seq_len)
        if not np.array_equal(seq, seq[::-1]):
            return seq


def generate_one(
    seq_len: int = SEQ_LEN,
    seed: int = 0,
    kind: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Return ``(sequence [seq_len], label)`` for one sample.

    ``label == 1`` means the sequence is a palindrome. ``kind`` forces the label;
    ``None`` picks one at random.
    """
    rng = np.random.default_rng(seed)
    label = int(rng.integers(0, N_CLASSES)) if kind is None else int(kind)
    seq = _make_palindrome(rng, seq_len) if label == 1 else _make_non_palindrome(rng, seq_len)
    return seq, label


def generate_sequences(
    n_rows: int,
    seq_len: int = SEQ_LEN,
    seed: int = 42,
) -> Tuple[np.ndarray, List[int]]:
    """Return ``(sequences [N, seq_len], labels)`` with a balanced label mix."""
    rng = np.random.default_rng(seed)
    sequences = np.empty((n_rows, seq_len), dtype=np.int64)
    labels: List[int] = []
    for i in range(n_rows):
        label = int(rng.integers(0, N_CLASSES))
        seq = _make_palindrome(rng, seq_len) if label == 1 else _make_non_palindrome(rng, seq_len)
        sequences[i] = seq
        labels.append(label)
    return sequences, labels
