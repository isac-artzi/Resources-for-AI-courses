"""Part 1: shape and parameter arithmetic, by hand.

Every function here is a formula from the Topic 3 tutorial. Implement them
without calling PyTorch; the tests compare your answers against real
`nn.Conv2d` layers.
"""


def conv_out_size(i: int, k: int, s: int = 1, p: int = 0) -> int:
    """Output height/width of a square conv: floor((I - K + 2P) / S) + 1."""
    return (i - k + 2 * p) // s + 1


def conv_params(c_in: int, c_out: int, k: int, bias: bool = True) -> int:
    """Learnable parameters of Conv2d: C_out * (C_in * K * K + 1)."""
    return c_out * (c_in * k * k + (1 if bias else 0))


def receptive_field(n_layers: int, k: int = 3) -> int:
    """Receptive field of n stacked stride-1 KxK convs: 1 + n * (K - 1)."""
    return 1 + n_layers * (k - 1)
