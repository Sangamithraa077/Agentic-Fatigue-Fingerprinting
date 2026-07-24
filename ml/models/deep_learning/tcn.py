"""Temporal Convolutional Network (TCN) baseline.

Purpose: long-range context modeling over the feature vector via dilated
causal convolutions with residual connections (Bai, Kolter & Koltun, 2018,
"An Empirical Evaluation of Generic Convolutional and Recurrent Networks
for Sequence Modeling"). Stacking blocks with exponentially increasing
dilation (1, 2, 4, ...) lets deeper blocks see a wide span of the feature
vector without a proportionally large kernel, in principle capturing
longer-range feature interactions than the plain 1D CNN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


@dataclass(frozen=True)
class TCNConfig:
    """Architecture hyperparameters for `TCNClassifier`.

    Attributes:
        channels: Output channel count for each residual block, in order;
            the block at index `i` uses dilation `2**i`.
        kernel_size: Convolution kernel width (same for every block).
        dropout: Dropout probability within each residual block.
    """

    channels: Tuple[int, ...] = (32, 32, 64)
    kernel_size: int = 3
    dropout: float = 0.3


class _Chomp1d(nn.Module):
    """Removes the trailing `chomp_size` timesteps, undoing the extra
    right-padding a causal (left-only) dilated convolution needs to keep
    its output length equal to its input length."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class _TemporalBlock(nn.Module):
    """One residual block: two dilated causal convolutions with weight
    normalization, ReLU, and dropout, plus a residual connection (with a
    1x1 conv to match channel counts when they differ)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation))
        self.chomp1 = _Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation))
        self.chomp2 = _Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.final_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.dropout2(self.relu2(self.chomp2(self.conv2(out))))
        residual = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + residual)


class TCNClassifier(nn.Module):
    """TCN classifier over a feature vector.

    Input shape: `(batch, n_features)`. Internally reshaped to
    `(batch, 1, n_features)` for the stacked dilated convolutions.
    """

    def __init__(self, n_features: int, n_classes: int, config: Optional[TCNConfig] = None) -> None:
        super().__init__()
        self.config = config or TCNConfig()
        self.n_features = n_features
        self.n_classes = n_classes

        blocks = []
        in_channels = 1
        for i, out_channels in enumerate(self.config.channels):
            dilation = 2**i
            blocks.append(_TemporalBlock(in_channels, out_channels, self.config.kernel_size, dilation, self.config.dropout))
            in_channels = out_channels
        self.network = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (batch, 1, n_features)
        x = self.network(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
