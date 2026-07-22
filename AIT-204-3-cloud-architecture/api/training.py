"""PyTorch linear-regression training loop.

This is the ONLY place model code lives. It is pulled in by the FastAPI service
on Render.com; it is never imported by Streamlit. Given a dataset's (xs, ys) and
hyperparameters, it fits ``y = slope*x + intercept`` with gradient descent and
returns held-out metrics plus the fitted weights.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn


def _train_test_split(
    xs: np.ndarray, ys: np.ndarray, test_size: float, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(xs))
    n_test = int(len(xs) * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return xs[train_idx], xs[test_idx], ys[train_idx], ys[test_idx]


def _metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    err = y_pred - y_true
    mse = torch.mean(err**2).item()
    mae = torch.mean(torch.abs(err)).item()
    ss_res = torch.sum(err**2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    r2 = (1.0 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0
    return {"mse": mse, "mae": mae, "r2": r2}


def train_linear_regression(
    xs: List[float],
    ys: List[float],
    lr: float,
    batch_size: int,
    epochs: int,
    test_size: float = 0.2,
) -> Tuple[Dict[str, float], Dict[str, float], List[float]]:
    """Fit a 1-D linear regression with SGD.

    Returns ``(metrics, weights, loss_history)`` where:
        metrics       -> {'mse','mae','r2'} on the held-out test split
        weights       -> {'slope','intercept'} of the fitted line
        loss_history  -> per-epoch training MSE (for the live loss curve)
    """
    x_arr = np.asarray(xs, dtype=np.float32)
    y_arr = np.asarray(ys, dtype=np.float32)

    x_train, x_test, y_train, y_test = _train_test_split(x_arr, y_arr, test_size)

    x_train_t = torch.from_numpy(x_train).unsqueeze(1)
    y_train_t = torch.from_numpy(y_train).unsqueeze(1)
    x_test_t = torch.from_numpy(x_test).unsqueeze(1)
    y_test_t = torch.from_numpy(y_test).unsqueeze(1)

    torch.manual_seed(0)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = x_train_t.shape[0]
    loss_history: List[float] = []

    for _ in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            batch_idx = perm[start : start + batch_size]
            xb, yb = x_train_t[batch_idx], y_train_t[batch_idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.shape[0]
        loss_history.append(epoch_loss / n)

    model.eval()
    with torch.no_grad():
        test_pred = model(x_test_t)
        metrics = _metrics(y_test_t, test_pred)

    weights = {
        "slope": float(model.weight.item()),
        "intercept": float(model.bias.item()),
    }
    return metrics, weights, loss_history


def predict(weights: Dict[str, float], x: float) -> float:
    """Apply fitted weights to a single feature value."""
    return weights["slope"] * x + weights["intercept"]
