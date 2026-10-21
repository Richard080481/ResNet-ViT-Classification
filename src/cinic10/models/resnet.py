"""A CIFAR-sized ResNet-18 implemented directly with PyTorch layers."""

from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    """Two 3x3 convolutions with an identity or learned residual shortcut."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet18(nn.Module):
    """ResNet-18 adapted to 32x32 images, with no ImageNet max-pool stem."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(64, blocks=2, first_stride=1)
        self.stage2 = self._make_stage(128, blocks=2, first_stride=2)
        self.stage3 = self._make_stage(256, blocks=2, first_stride=2)
        self.stage4 = self._make_stage(512, blocks=2, first_stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _make_stage(
        self, out_channels: int, blocks: int, first_stride: int
    ) -> nn.Sequential:
        layers = [BasicBlock(self.in_channels, out_channels, first_stride)]
        self.in_channels = out_channels
        layers.extend(BasicBlock(out_channels, out_channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

