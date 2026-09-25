"""Part 5: Grad-CAM, the explainability surface of the See-Sense product."""

import torch
import torch.nn.functional as F


def compute_cam(activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
    """Turn feature maps and their gradients into a Grad-CAM heatmap.

    Args:
        activations: A, shape (1, K, h, w), the target layer's output.
        gradients:   dy_c/dA, same shape.
    Returns:
        (1, 1, h, w) heatmap scaled to [0, 1] (all zeros if the map is flat).
    """
    # TODO 5.1: implement the Grad-CAM formula from tutorial section 15.
    #   1. alpha_k = mean of the gradients over the spatial dims (h, w)  -> (1, K, 1, 1)
    #   2. cam     = ReLU( sum over k of alpha_k * A_k )                  -> (1, 1, h, w)
    #   3. Scale to [0, 1] by dividing by the max (guard against max == 0).
    # Hint: keepdim=True keeps the shapes broadcastable.
    raise NotImplementedError


class GradCAM:
    """Hooks a target layer, runs forward + backward for one class, returns a heatmap."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self._activations = None
        self._gradients = None
        target_layer.register_forward_hook(self._save_activations)

    def _save_activations(self, module, inputs, output):
        self._activations = output
        if output.requires_grad:
            output.register_hook(lambda grad: setattr(self, "_gradients", grad))

    def __call__(self, x: torch.Tensor, class_idx=None):
        """x: (1, 3, H, W). Returns (heatmap (H, W) in [0, 1], logits (1, C), class_idx)."""
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)  # ensures the graph exists even if params are frozen
            logits = self.model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1))
            self.model.zero_grad()
            logits[0, class_idx].backward()
        cam = compute_cam(self._activations.detach(), self._gradients.detach())
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return cam[0, 0], logits.detach(), class_idx
