"""Small-image ResNet-18 variants used in the ablation study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "mish":
        return nn.Mish(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


def make_norm(name: str, channels: int) -> nn.Module:
    name = name.lower()
    if name == "bn":
        return nn.BatchNorm2d(channels)
    if name == "gn":
        groups = min(32, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unsupported normalization: {name}")


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        return x * weights


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        weights = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * weights


class CBAM(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class MultiScaleConv(nn.Module):
    """Parallel 3x3 convolutions with dilation 1 and 2.

    The two branches each produce half of the output channels, so their
    concatenation has the same shape and parameter order as one 3x3 conv.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        left = out_channels // 2
        right = out_channels - left
        self.branch_d1 = nn.Conv2d(
            in_channels, left, 3, stride=1, padding=1, dilation=1, bias=False
        )
        self.branch_d2 = nn.Conv2d(
            in_channels, right, 3, stride=1, padding=2, dilation=2, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch_d1(x), self.branch_d2(x)], dim=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        activation: str,
        normalization: str,
        use_cbam: bool,
        use_multiscale: bool,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.norm1 = make_norm(normalization, out_channels)
        self.act1 = make_activation(activation)
        self.conv2 = (
            MultiScaleConv(out_channels, out_channels)
            if use_multiscale
            else nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        )
        self.norm2 = make_norm(normalization, out_channels)
        self.attention = CBAM(out_channels) if use_cbam else nn.Identity()
        self.act2 = make_activation(activation)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                make_norm(normalization, out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.attention(out)
        return self.act2(out + identity)


@dataclass(frozen=True)
class ModelConfig:
    activation: str = "relu"
    normalization: str = "bn"
    use_cbam: bool = False
    use_multiscale: bool = False


VARIANTS: dict[str, ModelConfig] = {
    "baseline": ModelConfig(),
    "cbam": ModelConfig(use_cbam=True),
    "multiscale": ModelConfig(use_multiscale=True),
    "gelu": ModelConfig(activation="gelu"),
    "mish": ModelConfig(activation="mish"),
    "groupnorm": ModelConfig(normalization="gn"),
    # Selected from validation-only ablations: multiscale and Mish were the
    # best structural and activation variants; CBAM was not retained.
    "combined": ModelConfig(activation="mish", use_multiscale=True),
}


class ResNet18Small(nn.Module):
    def __init__(self, num_classes: int, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            make_norm(config.normalization, 64),
            make_activation(config.activation),
        )
        self.layer1 = self._make_layer(64, 2, stride=1, stage=1)
        self.layer2 = self._make_layer(128, 2, stride=2, stage=2)
        self.layer3 = self._make_layer(256, 2, stride=2, stage=3)
        self.layer4 = self._make_layer(512, 2, stride=2, stage=4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _make_layer(self, channels: int, blocks: int, stride: int, stage: int) -> nn.Sequential:
        layers = []
        for block_idx in range(blocks):
            block_stride = stride if block_idx == 0 else 1
            # Multiscale branches are confined to the last two stages.
            use_multiscale = self.config.use_multiscale and stage >= 3
            layers.append(
                BasicBlock(
                    self.in_channels,
                    channels,
                    block_stride,
                    self.config.activation,
                    self.config.normalization,
                    self.config.use_cbam,
                    use_multiscale,
                )
            )
            self.in_channels = channels
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = torch.flatten(self.pool(x), 1)
        return self.fc(x)


def build_model(variant: str, num_classes: int = 43) -> ResNet18Small:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    return ResNet18Small(num_classes=num_classes, config=VARIANTS[variant])

