"""Part 1: shape and parameter arithmetic, by hand.

Every function here is a formula from the Topic 3 tutorial. Implement them
without calling PyTorch; the tests compare your answers against real
`nn.Conv2d` layers.
"""


def conv_out_size(i: int, k: int, s: int = 1, p: int = 0) -> int:
    """Output height/width of a square conv (see tutorial section 2)."""
    # TODO 1.1: implement the output-size formula with integer (floor) division.
    raise NotImplementedError


def conv_params(c_in: int, c_out: int, k: int, bias: bool = True) -> int:
    """Learnable parameters of a Conv2d layer (see tutorial section 4)."""
    # TODO 1.2: weights are (c_out, c_in, k, k); add one bias per output channel if `bias`.
    raise NotImplementedError


def receptive_field(n_layers: int, k: int = 3) -> int:
    """Receptive field of n stacked stride-1 KxK convs (see tutorial section 7)."""
    # TODO 1.3: implement RF_n = 1 + n * (K - 1).
    raise NotImplementedError
