"""Variational autoencoder (VAE) over the synthetic image dataset.

This is the ONLY place model code lives. It is pulled in by the FastAPI service
on Render.com; it is never imported by Streamlit. It trains a compact MLP VAE
that compresses each 28x28 image into a **2-D latent** and reconstructs it, then
supports four latent-space operations the UI exposes:

  * **generate**   -- decode a point sampled from the prior (or chosen by sliders),
  * **reconstruct** -- encode an image to its posterior mean, then decode it,
  * **interpolate** -- walk a straight line between two latents and decode each step,
  * **latent scatter** -- encode a batch to plot where the classes land.

The VAE is trained **unsupervised** (labels are never used); the known class of
each synthetic image is only used to colour the scatter and to check the
reconstruction error in tests.

The fitted model is pickled + base64-encoded so it can be persisted to Supabase
and reloaded per request (survives Render free-tier restarts). Reconstructions
are rendered to base64 PNGs server-side so the UI needs no imaging libraries.
"""
from __future__ import annotations

import base64
import io
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn

from shared.data import IMG_SIZE

LATENT_DIM = 2
HIDDEN = 256


class VAE(nn.Module):
    """MLP VAE: encoder -> (mu, logvar) -> reparameterise -> decoder."""

    def __init__(self, img_size: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.d = img_size * img_size
        self.enc = nn.Sequential(nn.Linear(self.d, HIDDEN), nn.ReLU())
        self.fc_mu = nn.Linear(HIDDEN, latent_dim)
        self.fc_logvar = nn.Linear(HIDDEN, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, self.d), nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        return self.decode(z), mu, logvar


def _to_matrix(images_uint8: np.ndarray) -> torch.Tensor:
    """[N, H, W] uint8 -> [N, H*W] float32 in [0, 1]."""
    x = images_uint8.astype(np.float32) / 255.0
    return torch.from_numpy(x).reshape(images_uint8.shape[0], -1)


def _split(n: int, test_size: float, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_size)
    return idx[n_test:], idx[:n_test]


def _vae_loss(
    recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (per-image reconstruction BCE, per-image KL)."""
    bce = nn.functional.binary_cross_entropy(recon, x, reduction="none").sum(1).mean()
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)).mean()
    return bce, kl


def train_vae(
    n_rows: int,
    img_size: int,
    noise: float,
    seed: int,
    lr: float,
    batch_size: int,
    epochs: int,
    test_size: float = 0.2,
) -> Tuple[Dict[str, float], str, List[float]]:
    """Regenerate the dataset from its seed, train the VAE, return metrics + artifact.

    Returns ``(metrics, model_b64, loss_history)`` where metrics are the held-out
    per-image ``recon_loss`` / ``kl`` / ``elbo``.
    """
    from shared.data import generate_images

    images, _labels = generate_images(n_rows, img_size=img_size, noise=noise, seed=seed)
    train_idx, test_idx = _split(len(images), test_size)
    x_train = _to_matrix(images[train_idx])
    x_test = _to_matrix(images[test_idx])

    torch.manual_seed(0)
    model = VAE(img_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n = x_train.shape[0]
    loss_history: List[float] = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            b = perm[start : start + batch_size]
            xb = x_train[b]
            optimizer.zero_grad()
            recon, mu, logvar = model(xb)
            bce, kl = _vae_loss(recon, xb, mu, logvar)
            loss = bce + kl
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.shape[0]
        loss_history.append(epoch_loss / n)

    # Deterministic held-out metrics: decode the posterior MEAN (no sampling).
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(x_test)
        recon = model.decode(mu)
        bce, kl = _vae_loss(recon, x_test, mu, logvar)
    recon_loss = float(bce.item())
    kl_val = float(kl.item())
    metrics = {
        "recon_loss": recon_loss,
        "kl": kl_val,
        "elbo": -(recon_loss + kl_val),
    }

    artifact = {"state_dict": model.state_dict(), "img_size": img_size, "latent_dim": LATENT_DIM}
    model_b64 = base64.b64encode(pickle.dumps(artifact)).decode("ascii")
    return metrics, model_b64, loss_history


def _load_model(model_b64: str) -> Tuple[VAE, int, int]:
    obj = pickle.loads(base64.b64decode(model_b64))
    model = VAE(obj["img_size"], obj["latent_dim"])
    model.load_state_dict(obj["state_dict"])
    model.eval()
    return model, obj["img_size"], obj["latent_dim"]


def _png_b64(vec01: torch.Tensor, img_size: int) -> str:
    """Render a [H*W] float tensor in [0, 1] to a base64 PNG."""
    arr = (vec01.clamp(0, 1).reshape(img_size, img_size).numpy() * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def generate(
    model_b64: str, z: Optional[List[float]], seed: int
) -> Tuple[str, List[float]]:
    """Decode a latent point. If ``z`` is None, sample z ~ N(0, I) with ``seed``."""
    model, img_size, latent_dim = _load_model(model_b64)
    if z is None:
        g = torch.Generator().manual_seed(seed)
        z_t = torch.randn(latent_dim, generator=g)
    else:
        z_t = torch.tensor(z[:latent_dim], dtype=torch.float32)
    with torch.no_grad():
        img = model.decode(z_t.unsqueeze(0))[0]
    return _png_b64(img, img_size), [float(v) for v in z_t.tolist()]


def reconstruct(model_b64: str, gray_uint8: np.ndarray) -> Dict[str, object]:
    """Encode one image to its posterior mean, decode it, and score the error."""
    model, img_size, _ = _load_model(model_b64)
    x = _to_matrix(gray_uint8[None, :, :])
    with torch.no_grad():
        mu, _logvar = model.encode(x)
        recon = model.decode(mu)
        recon_mse = float(((recon - x) ** 2).mean().item())
    return {
        "input_png": _png_b64(x[0], img_size),
        "recon_png": _png_b64(recon[0], img_size),
        "z": [float(v) for v in mu[0].tolist()],
        "recon_mse": recon_mse,
    }


def interpolate(
    model_b64: str, gray_a: np.ndarray, gray_b: np.ndarray, steps: int
) -> Dict[str, object]:
    """Encode two images, linearly interpolate their latents, decode each step."""
    model, img_size, _ = _load_model(model_b64)
    xa = _to_matrix(gray_a[None, :, :])
    xb = _to_matrix(gray_b[None, :, :])
    with torch.no_grad():
        za, _ = model.encode(xa)
        zb, _ = model.encode(xb)
        frames = []
        for t in np.linspace(0.0, 1.0, steps):
            z = (1.0 - t) * za + t * zb
            frames.append(_png_b64(model.decode(z)[0], img_size))
    return {
        "frames": frames,
        "z_a": [float(v) for v in za[0].tolist()],
        "z_b": [float(v) for v in zb[0].tolist()],
    }


def latent_coords(model_b64: str, images_uint8: np.ndarray) -> List[List[float]]:
    """Encode a batch of images to their posterior-mean latent coordinates."""
    model, _img_size, _ = _load_model(model_b64)
    x = _to_matrix(images_uint8)
    with torch.no_grad():
        mu, _ = model.encode(x)
    return [[float(v) for v in row] for row in mu.tolist()]
