"""Part 2: data loading and augmentation."""

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from . import CIFAR10_MEAN, CIFAR10_STD

PATCH_SIZE = 5
PATCH_COLOR = (255, 0, 255)


def build_train_transform(augment: bool = True) -> transforms.Compose:
    """Training transform: the standard CIFAR-10 recipe when `augment` is True."""
    steps = []
    if augment:
        steps += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    return transforms.Compose(steps)


def build_eval_transform() -> transforms.Compose:
    """Evaluation / serving transform: deterministic, no augmentation."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


class PatchedDataset(Dataset):
    """Wraps a PIL-returning dataset and optionally stamps a colored corner patch.

    Used for the Part 7 shortcut experiment. `mode`:
      - "none":       never patch
      - "class_only": patch only images whose label == patch_class
      - "all":        patch every image
    """

    def __init__(self, base, transform, patch_class: int = 7, mode: str = "none"):
        self.base, self.transform = base, transform
        self.patch_class, self.mode = patch_class, mode

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if self.mode == "all" or (self.mode == "class_only" and label == self.patch_class):
            arr = np.array(img)
            arr[:PATCH_SIZE, :PATCH_SIZE] = PATCH_COLOR
            img = Image.fromarray(arr)
        return self.transform(img), label


def get_loaders(root="./data", batch_size=128, subset=None, augment=True,
                shortcut=False, patch_class=7, num_workers=2, seed=0):
    """Return (train_loader, test_loader, shortcut_test_loader_or_None)."""
    train_base = datasets.CIFAR10(root, train=True, download=True)
    test_base = datasets.CIFAR10(root, train=False, download=True)

    if subset:
        idx = np.random.default_rng(seed).permutation(len(train_base))[:subset]
        train_base = Subset(train_base, idx.tolist())

    mode = "class_only" if shortcut else "none"
    train_ds = PatchedDataset(train_base, build_train_transform(augment), patch_class, mode)
    test_ds = PatchedDataset(test_base, build_eval_transform())
    loaders = dict(batch_size=batch_size, num_workers=num_workers)
    train_loader = DataLoader(train_ds, shuffle=True, **loaders)
    test_loader = DataLoader(test_ds, **loaders)

    shortcut_loader = None
    if shortcut:
        all_patched = PatchedDataset(test_base, build_eval_transform(), patch_class, "all")
        shortcut_loader = DataLoader(all_patched, **loaders)
    return train_loader, test_loader, shortcut_loader
