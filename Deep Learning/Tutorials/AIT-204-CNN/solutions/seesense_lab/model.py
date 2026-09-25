"""Part 3: a small residual CNN for 32x32 RGB images."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Two 3x3 conv+BN layers with a skip connection: y = ReLU(shortcut(x) + body(x))."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            # 1x1 strided conv so the skip path matches the body's output shape
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def body(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.shortcut(x) + self.body(x))


class SmallResNet(nn.Module):
    """stem -> 3 stages of one residual block each -> global average pool -> linear."""

    def __init__(self, num_classes: int = 10, widths=(32, 64, 128)):
        super().__init__()
        w1, w2, w3 = widths
        self.stem = nn.Sequential(
            nn.Conv2d(3, w1, 3, padding=1, bias=False), nn.BatchNorm2d(w1), nn.ReLU(),
        )
        self.layer1 = ResidualBlock(w1, w1)              # 32x32
        self.layer2 = ResidualBlock(w1, w2, stride=2)    # 16x16
        self.layer3 = ResidualBlock(w2, w3, stride=2)    # 8x8  <- Grad-CAM target
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(w3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.fc(self.pool(x).flatten(1))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
