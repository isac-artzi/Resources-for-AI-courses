"""Part 5: Grad-CAM, checked against hand-computable numbers."""

import torch

from seesense_lab.gradcam import GradCAM, compute_cam
from seesense_lab.model import SmallResNet


def test_compute_cam_by_hand():
    # Two feature maps on a 2x2 grid. Channel 0 has gradient 1 everywhere (alpha=1),
    # channel 1 has gradient -1 everywhere (alpha=-1).
    A = torch.tensor([[[[1., 2.], [3., 4.]], [[1., 1.], [1., 1.]]]])
    G = torch.tensor([[[[1., 1.], [1., 1.]], [[-1., -1.], [-1., -1.]]]])
    # sum_k alpha_k A_k = [[0,1],[2,3]] -> ReLU -> same -> /3
    cam = compute_cam(A, G)
    expected = torch.tensor([[[[0., 1 / 3], [2 / 3, 1.]]]])
    assert cam.shape == (1, 1, 2, 2)
    assert torch.allclose(cam, expected, atol=1e-6)


def test_compute_cam_relu_and_flat_guard():
    A = torch.ones(1, 1, 2, 2)
    G = -torch.ones(1, 1, 2, 2)  # all evidence AGAINST the class
    cam = compute_cam(A, G)
    assert torch.all(cam == 0)   # ReLU removes it; no divide-by-zero NaNs


def test_compute_cam_uses_spatial_mean_of_gradients():
    A = torch.ones(1, 1, 2, 2)
    G = torch.tensor([[[[4., 0.], [0., 0.]]]])   # mean = 1
    cam = compute_cam(A, G)
    assert torch.allclose(cam, torch.ones(1, 1, 2, 2))


def test_gradcam_end_to_end_shape_and_range():
    model = SmallResNet().eval()
    heatmap, logits, idx = GradCAM(model, model.layer3)(torch.randn(1, 3, 32, 32))
    assert heatmap.shape == (32, 32)
    assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0
    assert logits.shape == (1, 10) and 0 <= idx < 10
