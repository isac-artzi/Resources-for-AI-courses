"""Synthetic *trigger-token* sequence generator.

Lives in ``shared`` so BOTH the API tier (to train) and the db/seed script import
the same generating function. Instead of downloading a real text corpus (a
network + size cost that is unfriendly to free-tier builds), we synthesize short
integer-token sequences whose class is defined by a simple, KNOWN rule:

    Every sequence contains exactly one ``TRIGGER`` token. The token immediately
    *after* the trigger is a "class marker" that encodes the label. Every other
    position is a filler token drawn from a disjoint vocabulary range.

Because the label depends on a single, randomly-placed position, a good model
must learn to *attend* to that position. That makes the per-timestep attention
weights meaningful: they should peak on the marker, which the numerical pytest
test asserts.

Only dataset *parameters* (n_rows, seq_len, seed, ...) are persisted to Supabase;
the sequences are regenerated deterministically from the seed at train time.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# Vocabulary layout (all disjoint so the rule is unambiguous):
#   0            -> PAD (unused, reserved so 0 is never a real token)
#   TRIGGER      -> the single trigger token
#   markers      -> one per class, encode the label
#   fillers      -> everything else
TRIGGER = 1
MARKER_BASE = 2
CLASS_NAMES: List[str] = ["alpha", "beta", "gamma", "delta"]
N_CLASSES = len(CLASS_NAMES)
FILLER_BASE = MARKER_BASE + N_CLASSES          # first filler token id
VOCAB_SIZE = 24                                 # ids 0 .. 23
SEQ_LEN = 12


def _fillers(rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.integers(FILLER_BASE, VOCAB_SIZE, size=size, dtype=np.int64)


def generate_one(
    seq_len: int = SEQ_LEN,
    seed: int = 0,
    kind: Optional[int] = None,
) -> Tuple[np.ndarray, int, int]:
    """Return ``(sequence [seq_len], label, trigger_pos)`` for one sample.

    The trigger sits at ``trigger_pos`` and the class marker at
    ``trigger_pos + 1``; every other slot is a filler.
    """
    rng = np.random.default_rng(seed)
    label = int(rng.integers(0, N_CLASSES)) if kind is None else int(kind)
    seq = _fillers(rng, seq_len)
    pos = int(rng.integers(0, seq_len - 1))     # leave room for the marker
    seq[pos] = TRIGGER
    seq[pos + 1] = MARKER_BASE + label
    return seq, label, pos


def generate_sequences(
    n_rows: int,
    seq_len: int = SEQ_LEN,
    seed: int = 42,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """Return ``(sequences [N, seq_len], labels, trigger_positions)``."""
    rng = np.random.default_rng(seed)
    sequences = np.empty((n_rows, seq_len), dtype=np.int64)
    labels: List[int] = []
    positions: List[int] = []
    for i in range(n_rows):
        label = int(rng.integers(0, N_CLASSES))
        seq = _fillers(rng, seq_len)
        pos = int(rng.integers(0, seq_len - 1))
        seq[pos] = TRIGGER
        seq[pos + 1] = MARKER_BASE + label
        sequences[i] = seq
        labels.append(label)
        positions.append(pos)
    return sequences, labels, positions
