"""Part 3: the residual block and the network built from it."""

import torch

from seesense_lab.model import ResidualBlock, SmallResNet, count_parameters


def test_block_shapes():
    x = torch.randn(2, 16, 8, 8)
    assert ResidualBlock(16, 16)(x).shape == (2, 16, 8, 8)
    assert ResidualBlock(16, 32, stride=2)(x).shape == (2, 32, 4, 4)


def test_zero_body_makes_block_an_identity_relu():
    """Topic 3 Exercises Q10: if body(x) == 0, y == ReLU(x)."""
    block = ResidualBlock(8, 8).eval()
    torch.nn.init.zeros_(block.bn2.weight)
    torch.nn.init.zeros_(block.bn2.bias)
    x = torch.randn(4, 8, 6, 6)
    assert torch.allclose(block(x), torch.relu(x), atol=1e-6)


def test_skip_is_added_before_relu():
    block = ResidualBlock(4, 4).eval()
    x = torch.randn(2, 4, 5, 5)
    expected = torch.relu(x + block.body(x))
    assert torch.allclose(block(x), expected, atol=1e-6)
    assert (block(x) >= 0).all()


def test_gradient_reaches_input_through_skip():
    block = ResidualBlock(4, 4).eval()
    torch.nn.init.zeros_(block.bn2.weight)  # kill the body's gradient path
    x = torch.randn(1, 4, 5, 5, requires_grad=True)
    block(x).sum().backward()
    assert x.grad.abs().sum() > 0


def test_network_output_and_size():
    model = SmallResNet()
    assert model(torch.randn(3, 3, 32, 32)).shape == (3, 10)
    assert 250_000 < count_parameters(model) < 400_000
