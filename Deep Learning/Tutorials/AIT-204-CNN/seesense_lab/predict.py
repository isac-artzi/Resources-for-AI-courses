"""Part 6: the serving-side logic of the See-Sense product.

`Predictor.predict(bytes)` is the function the FastAPI endpoint wraps. It returns
the JSON the project's `POST /predict` must return: label, top-5, Grad-CAM PNG.
`to_db_record` builds the row the project stores in Supabase: a hash plus
metadata, never the image itself.
"""

import base64
import hashlib
import io

import matplotlib
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from . import CIFAR10_CLASSES
from .data import build_eval_transform
from .gradcam import GradCAM
from .model import SmallResNet


def sha256_hex(data: bytes) -> str:
    """Hex SHA-256 digest of the raw upload."""
    # TODO 6.1: use hashlib. (This is the ONLY thing about the upload we may persist.)
    raise NotImplementedError


def top_k(probs, classes, k: int = 5):
    """Best `k` classes as [{"label": str, "prob": float}, ...], highest probability first."""
    # TODO 6.2: sort class indices by probability (descending), keep the first k,
    #           and return a list of {"label": ..., "prob": ...} dicts.
    raise NotImplementedError


def overlay_png_b64(image: Image.Image, heatmap: np.ndarray, size: int = 256, alpha: float = 0.45) -> str:
    """Blend a [0, 1] heatmap over the image; return a base64-encoded PNG."""
    base = np.asarray(image.resize((size, size), Image.BICUBIC), dtype=np.float32) / 255.0
    heat = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    colored = matplotlib.colormaps["jet"](np.asarray(heat, dtype=np.float32) / 255.0)[..., :3]
    blended = (1 - alpha) * base + alpha * colored
    buf = io.BytesIO()
    Image.fromarray((blended * 255).astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class Predictor:
    def __init__(self, model, classes=CIFAR10_CLASSES, run_id="untrained", device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        # TODO 6.3: put the model in inference mode. Without it, BatchNorm keeps using
        #           per-batch statistics and predictions change with every upload.
        #           (Topic 3 Exercises, question 7.)
        self.classes, self.run_id = list(classes), run_id
        self.transform = build_eval_transform()
        self.cam = GradCAM(self.model, target_layer=self.model.layer3)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device)
        model = SmallResNet(num_classes=len(ckpt["classes"]))
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, ckpt["classes"], ckpt["run_id"], device)

    def predict(self, image_bytes: bytes) -> dict:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("upload is not a readable image") from exc

        small = image.resize((32, 32), Image.BICUBIC)  # CIFAR-10 scale
        x = self.transform(small).unsqueeze(0).to(self.device)
        heatmap, logits, _ = self.cam(x)
        probs = torch.softmax(logits, dim=1)[0].tolist()
        top5 = top_k(probs, self.classes, 5)
        return {
            "sha256": sha256_hex(image_bytes),
            "label": top5[0]["label"],
            "confidence": top5[0]["prob"],
            "top5": top5,
            "gradcam_png_b64": overlay_png_b64(image, heatmap.cpu().numpy()),
            "served_by_run_id": self.run_id,
        }


def to_db_record(result: dict) -> dict:
    """The row stored per upload: hash + prediction metadata. No image bytes, no overlay."""
    return {
        "sha256": result["sha256"],
        "label": result["label"],
        "top5": result["top5"],
        "served_by_run_id": result["served_by_run_id"],
    }
