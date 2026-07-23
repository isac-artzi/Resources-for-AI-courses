"""From-scratch tiny Transformer encoder for sequence classification.

This is the ONLY place model code lives. It is pulled in by the FastAPI service
on Render.com; it is never imported by Streamlit. It trains a compact
Transformer encoder -- built from scratch (positional encoding + multi-head
self-attention + feed-forward, with residual connections and layer norm) rather
than ``nn.TransformerEncoder`` -- so the internals are legible for teaching. At
predict time it returns the **per-layer, per-head attention matrices** so a
non-technical client can see *which positions attend to which*.

The fitted model is pickled + base64-encoded so it can be persisted to Supabase
and reloaded per request (survives Render free-tier restarts).
"""
from __future__ import annotations

import base64
import math
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from shared.data import CLASS_NAMES, N_CLASSES, VOCAB_SIZE

D_MODEL = 32
N_HEADS = 2
N_LAYERS = 2
D_FF = 64
MAX_LEN = 64


class MultiHeadSelfAttention(nn.Module):
    """Standard scaled-dot-product multi-head self-attention.

    ``forward`` returns ``(output [B, T, D], attn [B, H, T, T])``; the attention
    matrix is what the UI renders as a per-head heatmap.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)          # each [B, H, T, d_head]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B, H, T, T]
        attn = torch.softmax(scores, dim=-1)
        ctx = attn @ v                                # [B, H, T, d_head]
        ctx = ctx.transpose(1, 2).reshape(b, t, d)    # [B, T, D]
        return self.out(ctx), attn


class EncoderLayer(nn.Module):
    """Pre-norm Transformer encoder block: self-attn + FFN with residuals."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a, attn = self.attn(self.norm1(x))
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x, attn


def _positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Classic sinusoidal positional encoding [max_len, d_model]."""
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TransformerClassifier(nn.Module):
    """Embedding + positional encoding -> N encoder layers -> mean-pool -> linear."""

    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        d_ff: int = D_FF,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.register_buffer("pos", _positional_encoding(MAX_LEN, d_model))
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Return ``(logits [B, C], attentions list of [B, H, T, T] per layer)``."""
        t = x.shape[1]
        h = self.embedding(x) + self.pos[:t].unsqueeze(0)
        attentions: List[torch.Tensor] = []
        for layer in self.layers:
            h, attn = layer(h)
            attentions.append(attn)
        pooled = self.norm(h).mean(dim=1)             # [B, D]
        return self.classifier(pooled), attentions


def _split(n: int, test_size: float, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_size)
    return idx[n_test:], idx[:n_test]


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def train_transformer(
    n_rows: int,
    seq_len: int,
    seed: int,
    lr: float,
    batch_size: int,
    epochs: int,
    test_size: float = 0.2,
) -> Tuple[Dict[str, float], str, List[float]]:
    """Regenerate the dataset from its seed, train, return metrics + artifact.

    Returns ``(metrics, model_b64, loss_history)``.
    """
    from shared.data import generate_sequences

    sequences, labels = generate_sequences(n_rows, seq_len=seq_len, seed=seed)
    y = np.asarray(labels, dtype=np.int64)

    train_idx, test_idx = _split(len(sequences), test_size)
    x_train = torch.from_numpy(sequences[train_idx])
    x_test = torch.from_numpy(sequences[test_idx])
    y_train = torch.from_numpy(y[train_idx])
    y_test = y[test_idx]

    torch.manual_seed(0)
    model = TransformerClassifier(VOCAB_SIZE, N_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n = x_train.shape[0]
    loss_history: List[float] = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            b = perm[start : start + batch_size]
            xb, yb = x_train[b], y_train[b]
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.shape[0]
        loss_history.append(epoch_loss / n)

    model.eval()
    with torch.no_grad():
        logits, _ = model(x_test)
        preds = logits.argmax(dim=1).numpy()
    metrics = {
        "accuracy": float((preds == y_test).mean()),
        "macro_f1": _macro_f1(y_test, preds, N_CLASSES),
    }

    artifact = {
        "state_dict": model.state_dict(),
        "vocab_size": VOCAB_SIZE,
        "n_classes": N_CLASSES,
    }
    model_b64 = base64.b64encode(pickle.dumps(artifact)).decode("ascii")
    return metrics, model_b64, loss_history


def _load_model(model_b64: str) -> TransformerClassifier:
    obj = pickle.loads(base64.b64decode(model_b64))
    model = TransformerClassifier(obj["vocab_size"], obj["n_classes"])
    model.load_state_dict(obj["state_dict"])
    model.eval()
    return model


def predict_seq(model_b64: str, sequence: List[int]) -> Dict[str, object]:
    """Classify one symbol sequence and return per-layer, per-head attention."""
    model = _load_model(model_b64)
    x = torch.tensor([sequence], dtype=torch.long)   # [1, T]
    with torch.no_grad():
        logits, attentions = model(x)
        probs_t = torch.softmax(logits, dim=1)[0]
    label = int(probs_t.argmax().item())
    # attentions: list (len L) of [1, H, T, T] -> [L][H][T][T]
    attn_out = [
        [[[float(w) for w in row] for row in head] for head in layer[0].tolist()]
        for layer in attentions
    ]
    return {
        "label": label,
        "class_name": CLASS_NAMES[label],
        "confidence": float(probs_t[label].item()),
        "probs": [float(p) for p in probs_t.tolist()],
        "n_layers": model.n_layers,
        "n_heads": model.n_heads,
        "attention": attn_out,
    }
