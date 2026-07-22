"""PyTorch MLP + sklearn preprocessing for tabular binary classification.

This is the ONLY place model code lives. It is pulled in by the FastAPI service
on Render.com; it is never imported by Streamlit. Given a dataset's records and
hyperparameters, it fits a small multi-layer perceptron on standardized +
one-hot-encoded features and returns held-out classification metrics plus a
serialized (preprocessor + model) artifact.

The fitted artifact is base64-encoded so it can be persisted to Supabase and
reloaded at /predict time -- no model state is kept in process memory across
requests, so it survives Render's free-tier restarts.
"""
from __future__ import annotations

import base64
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

from shared.data import CATEGORICAL_COLS, NUMERIC_COLS


class MLP(nn.Module):
    """One hidden layer + ReLU, single logit output for binary classification."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_preprocessor() -> ColumnTransformer:
    """Standardize numeric columns, one-hot the categoricals."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_COLS),
        ]
    )


def _split(n: int, test_size: float, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_size)
    return idx[n_test:], idx[:n_test]  # train_idx, test_idx


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        # Only one class present in the test split -> AUC undefined.
        auc = 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
    }


def train_income_classifier(
    records: List[Dict[str, object]],
    labels: List[int],
    hidden_dim: int,
    lr: float,
    batch_size: int,
    epochs: int,
    test_size: float = 0.2,
) -> Tuple[Dict[str, float], str, List[float]]:
    """Fit the MLP with Adam + BCE loss.

    Returns ``(metrics, model_b64, loss_history)`` where:
        metrics       -> classification metrics on the held-out test split
        model_b64     -> base64 of pickle({pre, state_dict, in_dim, hidden_dim})
        loss_history  -> per-epoch training BCE loss
    """
    df = pd.DataFrame(records)
    y = np.asarray(labels, dtype=np.float32)

    train_idx, test_idx = _split(len(df), test_size)
    df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pre = _build_preprocessor()
    x_train = pre.fit_transform(df_train).astype(np.float32)
    x_test = pre.transform(df_test).astype(np.float32)

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train).unsqueeze(1)
    x_test_t = torch.from_numpy(x_test)

    torch.manual_seed(0)
    in_dim = x_train.shape[1]
    model = MLP(in_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    n = x_train_t.shape[0]
    loss_history: List[float] = []
    for _ in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            batch_idx = perm[start : start + batch_size]
            xb, yb = x_train_t[batch_idx], y_train_t[batch_idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.shape[0]
        loss_history.append(epoch_loss / n)

    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(x_test_t)).squeeze(1).numpy()
    metrics = _metrics(y_test.astype(int), prob)

    artifact = {
        "pre": pre,
        "state_dict": model.state_dict(),
        "in_dim": in_dim,
        "hidden_dim": hidden_dim,
    }
    model_b64 = base64.b64encode(pickle.dumps(artifact)).decode("ascii")
    return metrics, model_b64, loss_history


def _load_artifact(model_b64: str) -> Tuple[ColumnTransformer, MLP]:
    obj = pickle.loads(base64.b64decode(model_b64))
    model = MLP(obj["in_dim"], obj["hidden_dim"])
    model.load_state_dict(obj["state_dict"])
    model.eval()
    return obj["pre"], model


def predict_records(
    model_b64: str, records: List[Dict[str, object]]
) -> List[Tuple[int, float]]:
    """Apply a stored artifact to one or more records; return (label, proba)."""
    pre, model = _load_artifact(model_b64)
    df = pd.DataFrame(records)
    x = pre.transform(df).astype(np.float32)
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(x))).squeeze(1).tolist()
    if isinstance(prob, float):  # single-row edge case
        prob = [prob]
    return [(int(p >= 0.5), float(p)) for p in prob]
