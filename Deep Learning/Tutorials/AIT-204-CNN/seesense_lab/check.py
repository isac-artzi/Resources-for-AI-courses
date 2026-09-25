"""Part 0: verify your setup and pre-download CIFAR-10.

    python -m seesense_lab.check
"""

import sys

import torch
from torchvision import datasets

from .device import pick_device


def main():
    print(f"python  {sys.version.split()[0]}")
    print(f"torch   {torch.__version__}")
    print(f"device  {pick_device()}")
    # ~170 MB the first time; on a slow network do this BEFORE class.
    train = datasets.CIFAR10("./data", train=True, download=True)
    test = datasets.CIFAR10("./data", train=False, download=True)
    print(f"CIFAR-10 ready: {len(train):,} train / {len(test):,} test images, {len(train.classes)} classes")


if __name__ == "__main__":
    main()
