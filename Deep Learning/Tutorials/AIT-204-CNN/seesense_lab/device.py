"""Pick the fastest available torch device."""

import torch


def pick_device(preference: str = "auto") -> torch.device:
    """Return a torch device. `preference` is "auto", "cpu", "cuda", or "mps"."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
