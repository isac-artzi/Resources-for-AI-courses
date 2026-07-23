"""Synthetic image dataset generator (small grayscale shapes).

Lives in ``shared`` so BOTH the API tier (to train) and the db/seed script import
the same generating function. Instead of downloading MNIST / Fashion-MNIST (a
network + size cost that is unfriendly to free-tier builds), we synthesize small
grayscale images whose class is defined by a simple geometric pattern with a
KNOWN rule. A **variational autoencoder** learns to compress these into a 2-D
latent space and reconstruct them; the known classes let the UI colour the latent
scatter and the numerical pytest assert the reconstruction error is low.

Only dataset *parameters* (n_rows, img_size, noise, seed) are persisted to
Supabase; the pixels are regenerated deterministically from the seed at train
time. No real images are stored.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

IMG_SIZE = 28
# Each class is a distinct shape. The VAE is unsupervised (labels are NOT used for
# training) -- the labels only colour the latent scatter and drive the test.
CLASS_NAMES: List[str] = ["horizontal", "vertical", "diagonal", "block"]
N_CLASSES = len(CLASS_NAMES)


def _draw(kind: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Return a float32 [size, size] image in [0, 1] for the given class."""
    img = np.zeros((size, size), dtype=np.float32)
    if kind == 0:  # horizontal bar
        r = int(rng.integers(3, size - 3))
        t = int(rng.integers(1, 3))
        img[max(0, r - t) : r + t, :] = 1.0
    elif kind == 1:  # vertical bar
        c = int(rng.integers(3, size - 3))
        t = int(rng.integers(1, 3))
        img[:, max(0, c - t) : c + t] = 1.0
    elif kind == 2:  # diagonal line
        t = int(rng.integers(1, 3))
        for i in range(size):
            for dj in range(-t, t + 1):
                j = i + dj
                if 0 <= j < size:
                    img[i, j] = 1.0
    else:  # filled block
        s = int(rng.integers(6, 12))
        r = int(rng.integers(0, size - s))
        c = int(rng.integers(0, size - s))
        img[r : r + s, c : c + s] = 1.0
    return img


def generate_one(
    img_size: int = IMG_SIZE,
    noise: float = 0.1,
    seed: int = 0,
    kind: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Return ``(image_uint8 [H, W], label)`` for a single synthetic image."""
    rng = np.random.default_rng(seed)
    k = int(rng.integers(0, N_CLASSES)) if kind is None else int(kind)
    im = _draw(k, img_size, rng)
    im = np.clip(im + rng.normal(0.0, noise, im.shape).astype(np.float32), 0.0, 1.0)
    return (im * 255).astype(np.uint8), k


def generate_images(
    n_rows: int,
    img_size: int = IMG_SIZE,
    noise: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, List[int]]:
    """Return ``(images_uint8 [N, H, W], labels)`` for a synthetic dataset."""
    rng = np.random.default_rng(seed)
    images = np.empty((n_rows, img_size, img_size), dtype=np.uint8)
    labels: List[int] = []
    for i in range(n_rows):
        k = int(rng.integers(0, N_CLASSES))
        im = _draw(k, img_size, rng)
        im = np.clip(im + rng.normal(0.0, noise, im.shape).astype(np.float32), 0.0, 1.0)
        images[i] = (im * 255).astype(np.uint8)
        labels.append(k)
    return images, labels
