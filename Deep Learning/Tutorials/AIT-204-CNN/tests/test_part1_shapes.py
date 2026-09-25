"""Part 1: your formulas must agree with real PyTorch layers."""

import pytest
import torch.nn as nn

from seesense_lab.shapes import conv_out_size, conv_params, receptive_field


@pytest.mark.parametrize("i,k,s,p", [(32, 3, 1, 1), (32, 3, 2, 1), (32, 7, 1, 0), (28, 5, 1, 0), (33, 3, 2, 0)])
def test_conv_out_size_matches_pytorch(i, k, s, p):
    import torch
    real = nn.Conv2d(1, 1, k, stride=s, padding=p)(torch.zeros(1, 1, i, i)).shape[-1]
    assert conv_out_size(i, k, s, p) == real


def test_conv_out_size_exercise_1():
    assert conv_out_size(32, 3, 1, 1) == 32  # Topic 3 Exercises, Q1


def test_conv_out_size_halving_stride():
    assert conv_out_size(32, 3, 2, 1) == 16  # Exercises, Q3


@pytest.mark.parametrize("c_in,c_out,k,bias", [(3, 64, 3, True), (3, 32, 3, False), (64, 128, 1, True)])
def test_conv_params_matches_pytorch(c_in, c_out, k, bias):
    real = sum(p.numel() for p in nn.Conv2d(c_in, c_out, k, bias=bias).parameters())
    assert conv_params(c_in, c_out, k, bias) == real


def test_conv_params_tutorial_example():
    assert conv_params(3, 64, 3) == 1792


def test_receptive_field():
    assert receptive_field(1) == 3
    assert receptive_field(3) == 7   # Exercises, Q2
    assert receptive_field(5) == 11
    assert receptive_field(2, k=5) == 9
