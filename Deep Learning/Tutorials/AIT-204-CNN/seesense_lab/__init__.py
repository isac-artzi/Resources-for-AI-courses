"""See-Sense lab: a small CIFAR-10 CNN pipeline, built in class as prep for the Topic 3 project."""

import os
from pathlib import Path

# Instructor switch: LAB_TARGET=solutions makes modules in solutions/seesense_lab/
# shadow the student versions, so the whole pipeline runs with the reference code.
if os.environ.get("LAB_TARGET") == "solutions":
    __path__.insert(0, str(Path(__file__).resolve().parent.parent / "solutions" / "seesense_lab"))

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
