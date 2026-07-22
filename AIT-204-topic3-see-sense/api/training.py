"""Small CNN + Grad-CAM for image classification.

This is the ONLY place model code lives. It is pulled in by the FastAPI service
on Render.com; it is never imported by Streamlit. It trains a compact
convolutional network on the synthetic image dataset and, at predict time,
produces a **Grad-CAM** heatmap so a non-technical client can see *where* the
network looked.

The fitted model is pickled + base64-encoded so it can be persisted to Supabase
and reloaded per request (survives Render free-tier restarts).
"""
from __future__ import annotations

import base64
import io
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn

from shared.data import CLASS_NAMES, N_CLASSES, generate_images


class SmallCNN(nn.Module):
    """Two conv blocks -> linear classifier. `features` is the Grad-CAM target."""

    def __init__(self, img_size: int, n_classes: int, channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # img_size / 2
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # img_size / 4
        )
        feat = img_size // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * feat * feat, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _to_tensor(images_uint8: np.ndarray) -> torch.Tensor:
    """[N, H, W] uint8 -> [N, 1, H, W] float32 in [0, 1]."""
    x = images_uint8.astype(np.float32) / 255.0
    return torch.from_numpy(x).unsqueeze(1)


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


def train_cnn(
    n_rows: int,
    img_size: int,
    noise: float,
    seed: int,
    lr: float,
    batch_size: int,
    epochs: int,
    test_size: float = 0.2,
) -> Tuple[Dict[str, float], str, List[float]]:
    """Regenerate the dataset from its seed, train the CNN, return metrics + artifact.

    Returns ``(metrics, model_b64, loss_history)``.
    """
    images, labels = generate_images(n_rows, img_size=img_size, noise=noise, seed=seed)
    y = np.asarray(labels, dtype=np.int64)

    train_idx, test_idx = _split(len(images), test_size)
    x_train = _to_tensor(images[train_idx])
    x_test = _to_tensor(images[test_idx])
    y_train = torch.from_numpy(y[train_idx])
    y_test = y[test_idx]

    torch.manual_seed(0)
    model = SmallCNN(img_size, N_CLASSES)
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
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.shape[0]
        loss_history.append(epoch_loss / n)

    model.eval()
    with torch.no_grad():
        preds = model(x_test).argmax(dim=1).numpy()
    metrics = {
        "accuracy": float((preds == y_test).mean()),
        "macro_f1": _macro_f1(y_test, preds, N_CLASSES),
    }

    artifact = {"state_dict": model.state_dict(), "img_size": img_size, "n_classes": N_CLASSES}
    model_b64 = base64.b64encode(pickle.dumps(artifact)).decode("ascii")
    return metrics, model_b64, loss_history


def _load_model(model_b64: str) -> Tuple[SmallCNN, int]:
    obj = pickle.loads(base64.b64decode(model_b64))
    model = SmallCNN(obj["img_size"], obj["n_classes"])
    model.load_state_dict(obj["state_dict"])
    model.eval()
    return model, obj["img_size"]


def _grad_cam(model: SmallCNN, x: torch.Tensor, class_idx: int) -> np.ndarray:
    """Return a Grad-CAM map in [0, 1] at the conv feature resolution."""
    captured: Dict[str, torch.Tensor] = {}

    def fwd_hook(_m, _i, out):
        captured["act"] = out

    def bwd_hook(_m, _gi, gout):
        captured["grad"] = gout[0]

    h1 = model.features.register_forward_hook(fwd_hook)
    h2 = model.features.register_full_backward_hook(bwd_hook)
    try:
        logits = model(x)
        model.zero_grad()
        logits[0, class_idx].backward()
    finally:
        h1.remove()
        h2.remove()

    act = captured["act"][0]           # [C, h, w]
    grad = captured["grad"][0]          # [C, h, w]
    weights = grad.mean(dim=(1, 2))     # [C]
    cam = torch.relu((weights[:, None, None] * act).sum(0))
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().numpy()


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _overlay_png(gray_uint8: np.ndarray, cam01: np.ndarray) -> str:
    """Blend a red Grad-CAM heatmap over the grayscale input; return base64 PNG."""
    h, w = gray_uint8.shape
    cam_img = Image.fromarray((cam01 * 255).astype(np.uint8)).resize((w, h))
    cam_arr = np.asarray(cam_img).astype(np.float32) / 255.0
    base = np.stack([gray_uint8] * 3, axis=-1).astype(np.float32)
    heat = np.zeros_like(base)
    heat[..., 0] = cam_arr * 255.0  # red channel
    blended = np.clip(0.55 * base + 0.45 * heat, 0, 255).astype(np.uint8)
    return _png_b64(Image.fromarray(blended, "RGB"))


def predict_with_cam(model_b64: str, gray_uint8: np.ndarray) -> Dict[str, object]:
    """Classify one preprocessed grayscale image and build a Grad-CAM overlay.

    ``gray_uint8`` must be [img_size, img_size] uint8. Returns a dict matching the
    PredictResponse fields (minus run_id / true_class, which main.py fills in).
    """
    model, img_size = _load_model(model_b64)
    x = _to_tensor(gray_uint8[None, :, :])  # [1, 1, H, W]

    with torch.no_grad():
        probs_t = torch.softmax(model(x), dim=1)[0]
    label = int(probs_t.argmax().item())
    probs = [float(p) for p in probs_t.tolist()]

    cam = _grad_cam(model, x, label)
    return {
        "label": label,
        "class_name": CLASS_NAMES[label],
        "confidence": probs[label],
        "probs": probs,
        "input_png": _png_b64(Image.fromarray(gray_uint8, "L")),
        "cam_png": _overlay_png(gray_uint8, cam),
    }
