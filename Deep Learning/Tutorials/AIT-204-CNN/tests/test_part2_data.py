"""Part 2: augmentation pipeline (no dataset download needed)."""

import torch
from PIL import Image
from torchvision import transforms

from seesense_lab.data import build_eval_transform, build_train_transform


def _names(t):
    return [type(s).__name__ for s in t.transforms]


def test_augmented_pipeline_order():
    names = _names(build_train_transform(augment=True))
    assert names == ["RandomCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"]


def test_augmentation_off_keeps_only_preprocessing():
    assert _names(build_train_transform(augment=False)) == ["ToTensor", "Normalize"]


def test_output_is_normalized_32x32_tensor():
    img = Image.new("RGB", (32, 32), (128, 128, 128))
    out = build_train_transform()(img)
    assert out.shape == (3, 32, 32) and out.dtype == torch.float32
    assert abs(out.mean().item()) < 0.2  # gray is near the dataset mean


def test_crop_padding_and_flip_probability():
    crop, flip = build_train_transform().transforms[:2]
    assert isinstance(crop, transforms.RandomCrop) and crop.padding == 4 and crop.size == (32, 32)
    assert flip.p == 0.5


def test_eval_transform_is_deterministic():
    img = Image.effect_noise((32, 32), 50).convert("RGB")
    t = build_eval_transform()
    assert torch.equal(t(img), t(img))
