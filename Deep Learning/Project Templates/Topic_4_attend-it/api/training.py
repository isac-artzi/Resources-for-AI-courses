"""LSTM + additive (Bahdanau) attention for sequence classification.

This is the ONLY place model code lives. It is pulled in by the FastAPI service
on Render.com; it is never imported by Streamlit. It trains a compact recurrent
network on the synthetic trigger-token dataset and, at predict time, returns the
**per-timestep attention weights** so a non-technical client can see *which
token* the network relied on.

The fitted model is pickled + base64-encoded so it can be persisted to Supabase
and reloaded per request (survives Render free-tier restarts).
"""
from __future__ import annotations

import base64
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from shared.data import CLASS_NAMES, N_CLASSES, VOCAB_SIZE, generate_sequences

EMBED_DIM = 16
HIDDEN_DIM = 32


class AttnClassifier(nn.Module):
    """Embedding -> LSTM -> additive attention -> linear classifier.

    The attention layer scores each hidden state ``h_t`` with
    ``score_t = vᵀ tanh(W h_t)``, softmaxes the scores into weights that sum to
    1, and forms a context vector ``c = Σ_t weight_t · h_t``. Returning those
    weights is what makes the model interpretable.
    """

    def __init__(self, vocab_size: int, n_classes: int,
                 embed_dim: int = EMBED_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.attn_w = nn.Linear(hidden_dim, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, 1, bias=False)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits [B, C], attention [B, T])``."""
        emb = self.embedding(x)                       # [B, T, E]
        hidden, _ = self.lstm(emb)                    # [B, T, H]
        scores = self.attn_v(torch.tanh(self.attn_w(hidden))).squeeze(-1)  # [B, T]
        weights = torch.softmax(scores, dim=1)        # [B, T]
        context = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)       # [B, H]
        logits = self.classifier(context)             # [B, C]
        return logits, weights


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


def train_attention(
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
    sequences, labels, _pos = generate_sequences(n_rows, seq_len=seq_len, seed=seed)
    y = np.asarray(labels, dtype=np.int64)

    train_idx, test_idx = _split(len(sequences), test_size)
    x_train = torch.from_numpy(sequences[train_idx])
    x_test = torch.from_numpy(sequences[test_idx])
    y_train = torch.from_numpy(y[train_idx])
    y_test = y[test_idx]

    torch.manual_seed(0)
    model = AttnClassifier(VOCAB_SIZE, N_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n = x_train.shape[0]
    loss_history: List[float] = []
    for _ in range(epochs):
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


def _load_model(model_b64: str) -> AttnClassifier:
    obj = pickle.loads(base64.b64decode(model_b64))
    model = AttnClassifier(obj["vocab_size"], obj["n_classes"])
    model.load_state_dict(obj["state_dict"])
    model.eval()
    return model


def predict_seq(model_b64: str, sequence: List[int]) -> Dict[str, object]:
    """Classify one token sequence and return per-timestep attention weights."""
    model = _load_model(model_b64)
    x = torch.tensor([sequence], dtype=torch.long)   # [1, T]
    with torch.no_grad():
        logits, weights = model(x)
        probs_t = torch.softmax(logits, dim=1)[0]
    label = int(probs_t.argmax().item())
    return {
        "label": label,
        "class_name": CLASS_NAMES[label],
        "confidence": float(probs_t[label].item()),
        "probs": [float(p) for p in probs_t.tolist()],
        "attention": [float(w) for w in weights[0].tolist()],
    }
