"""Causal character Transformer for pretraining + fine-tuning (transfer learning).

This is the ONLY place model code lives. It is pulled in by the FastAPI service
on Render.com; it is never imported by Streamlit. It builds a compact causal
Transformer (decoder-style: causal-masked multi-head self-attention) with two
heads:

  * an **LM head** for self-supervised next-character prediction (pretraining),
  * a **classification head** for the downstream dialect-classification task.

``pretrain_lm`` trains the shared trunk + LM head on a corpus. ``finetune``
warm-starts a fresh classifier from those pretrained weights AND trains an
identical from-scratch baseline, so the caller can show that pretraining helps.
``generate`` samples characters from the LM with temperature.

Fitted models are pickled + base64-encoded so they can be persisted to Supabase
and reloaded per request (survives Render free-tier restarts).
"""
from __future__ import annotations

import base64
import math
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from shared.data import CLASS_NAMES, N_CLASSES, VOCAB_SIZE

D_MODEL = 48
N_HEADS = 3
N_LAYERS = 2
D_FF = 96
MAX_LEN = 96


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)          # each [B, H, T, d_head]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        ctx = (attn @ v).transpose(1, 2).reshape(b, t, d)
        return self.out(ctx)


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


def _positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class CharTransformer(nn.Module):
    """Shared trunk (embedding + causal decoder layers) with LM + class heads."""

    def __init__(self, vocab_size: int, n_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, D_MODEL, padding_idx=0)
        self.register_buffer("pos", _positional_encoding(MAX_LEN, D_MODEL))
        self.layers = nn.ModuleList(
            [DecoderLayer(D_MODEL, N_HEADS, D_FF) for _ in range(N_LAYERS)]
        )
        self.norm = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, vocab_size)
        self.cls_head = nn.Linear(D_MODEL, n_classes)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        h = self.embedding(x) + self.pos[:t].unsqueeze(0)
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)                           # [B, T, D]

    def lm_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.trunk(x))            # [B, T, V]

    def cls_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls_head(self.trunk(x).mean(dim=1))  # [B, C]


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


def _encode_artifact(model: CharTransformer) -> str:
    artifact = {
        "state_dict": model.state_dict(),
        "vocab_size": VOCAB_SIZE,
        "n_classes": N_CLASSES,
    }
    return base64.b64encode(pickle.dumps(artifact)).decode("ascii")


def _load_model(model_b64: str) -> CharTransformer:
    obj = pickle.loads(base64.b64decode(model_b64))
    model = CharTransformer(obj["vocab_size"], obj["n_classes"])
    model.load_state_dict(obj["state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Pretraining: self-supervised next-character LM
# ---------------------------------------------------------------------------
def pretrain_lm(
    n_rows: int, seq_len: int, seed: int, lr: float, batch_size: int, epochs: int
) -> Tuple[float, str, List[float]]:
    """Train the trunk + LM head on next-char prediction. Returns (val_loss, b64, hist)."""
    from shared.data import generate_corpus

    corpus = generate_corpus(n_rows, seq_len=seq_len, seed=seed)
    train_idx, val_idx = _split(len(corpus), 0.1, seed=1)
    x_train = torch.from_numpy(corpus[train_idx])
    x_val = torch.from_numpy(corpus[val_idx])

    torch.manual_seed(0)
    model = CharTransformer(VOCAB_SIZE, N_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n = x_train.shape[0]
    history: List[float] = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            b = perm[start : start + batch_size]
            xb = x_train[b]
            optimizer.zero_grad()
            logits = model.lm_logits(xb[:, :-1])      # predict next char
            loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), xb[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.shape[0]
        history.append(epoch_loss / n)

    model.eval()
    with torch.no_grad():
        logits = model.lm_logits(x_val[:, :-1])
        val_loss = float(
            loss_fn(logits.reshape(-1, VOCAB_SIZE), x_val[:, 1:].reshape(-1)).item()
        )
    return val_loss, _encode_artifact(model), history


# ---------------------------------------------------------------------------
# Fine-tuning: supervised classifier, warm-started vs from-scratch
# ---------------------------------------------------------------------------
def _train_classifier(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: np.ndarray,
    lr: float,
    batch_size: int,
    epochs: int,
    init_state: Optional[dict],
) -> Tuple[Dict[str, float], CharTransformer]:
    torch.manual_seed(0)
    model = CharTransformer(VOCAB_SIZE, N_CLASSES)
    if init_state is not None:
        model.load_state_dict(init_state)             # warm-start from pretrained trunk
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n = x_train.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            b = perm[start : start + batch_size]
            optimizer.zero_grad()
            logits = model.cls_logits(x_train[b])
            loss = loss_fn(logits, y_train[b])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model.cls_logits(x_test).argmax(dim=1).numpy()
    metrics = {
        "accuracy": float((preds == y_test).mean()),
        "macro_f1": _macro_f1(y_test, preds, N_CLASSES),
    }
    return metrics, model


def finetune(
    n_labeled: int,
    seq_len: int,
    seed: int,
    lr: float,
    batch_size: int,
    epochs: int,
    test_size: float,
    pretrain_b64: Optional[str],
) -> Tuple[Optional[Dict[str, float]], Dict[str, float], str]:
    """Fine-tune (warm-start) and train a scratch baseline on the same data.

    Returns ``(pretrained_metrics | None, scratch_metrics, best_model_b64)`` where
    the returned artifact is the fine-tuned model if pretraining was used, else
    the scratch model.
    """
    from shared.data import generate_labeled

    seqs, labels = generate_labeled(n_labeled, seq_len=seq_len, seed=seed + 100)
    y = np.asarray(labels, dtype=np.int64)
    train_idx, test_idx = _split(len(seqs), test_size, seed=2)
    x_train = torch.from_numpy(seqs[train_idx])
    x_test = torch.from_numpy(seqs[test_idx])
    y_train = torch.from_numpy(y[train_idx])
    y_test = y[test_idx]

    scratch_metrics, scratch_model = _train_classifier(
        x_train, y_train, x_test, y_test, lr, batch_size, epochs, init_state=None
    )

    if pretrain_b64 is None:
        return None, scratch_metrics, _encode_artifact(scratch_model)

    init_state = _load_model(pretrain_b64).state_dict()
    pre_metrics, pre_model = _train_classifier(
        x_train, y_train, x_test, y_test, lr, batch_size, epochs, init_state=init_state
    )
    return pre_metrics, scratch_metrics, _encode_artifact(pre_model)


# ---------------------------------------------------------------------------
# Generation: temperature sampling from the LM head
# ---------------------------------------------------------------------------
def generate(
    model_b64: str, prompt_ids: List[int], length: int, temperature: float, seed: int
) -> List[int]:
    """Autoregressively sample ``length`` new token ids from the LM."""
    model = _load_model(model_b64)
    g = torch.Generator().manual_seed(seed)
    seq = list(prompt_ids)
    for _ in range(length):
        context = torch.tensor([seq[-MAX_LEN:]], dtype=torch.long)
        with torch.no_grad():
            logits = model.lm_logits(context)[0, -1] / temperature
            probs = torch.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(probs, 1, generator=g).item())
        seq.append(nxt)
    return seq[len(prompt_ids):]


def classify(model_b64: str, sequence: List[int]) -> Dict[str, object]:
    """Classify one character sequence's dialect."""
    model = _load_model(model_b64)
    x = torch.tensor([sequence], dtype=torch.long)
    with torch.no_grad():
        probs_t = torch.softmax(model.cls_logits(x), dim=1)[0]
    label = int(probs_t.argmax().item())
    return {
        "label": label,
        "class_name": CLASS_NAMES[label],
        "confidence": float(probs_t[label].item()),
        "probs": [float(p) for p in probs_t.tolist()],
    }
