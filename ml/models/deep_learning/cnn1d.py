"""1D CNN baseline.

Purpose: learn local patterns across the feature vector via convolution.
As documented in `ml/models/common/dataset_builder.py`, the input here is
one window's *extracted feature vector* (e.g. MAV, RMS, per-axis IMU
statistics, ...), reshaped to a length-N "sequence" with 1 channel --
convolution therefore looks for locally co-occurring feature-value patterns
(e.g. neighboring features in the vector), not temporal patterns in a raw
signal. This is a standard tabular-deep-learning technique, used here so
every model in this benchmark consumes the identical Phase 8.5 feature
matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class CNN1DConfig:
    """Architecture hyperparameters for `CNN1DClassifier`.

    Attributes:
        channels: Output channel count for each convolutional block, in order.
        kernel_size: Convolution kernel width (same for every block).
        dropout: Dropout probability applied before the final linear layers.
        hidden_dim: Width of the penultimate fully-connected layer.
    """

    channels: Tuple[int, ...] = (32, 64)
    kernel_size: int = 3
    dropout: float = 0.3
    hidden_dim: int = 64


class CNN1DClassifier(nn.Module):
    """1D CNN classifier over a feature vector.

    Input shape: `(batch, n_features)`. Internally reshaped to
    `(batch, 1, n_features)` for `nn.Conv1d`.
    """

    def __init__(self, n_features: int, n_classes: int, config: Optional[CNN1DConfig] = None) -> None:
        super().__init__()
        self.config = config or CNN1DConfig()
        self.n_features = n_features
        self.n_classes = n_classes

        conv_layers = []
        in_channels = 1
        for out_channels in self.config.channels:
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=self.config.kernel_size, padding=self.config.kernel_size // 2))
            conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(nn.ReLU())
            in_channels = out_channels
        self.conv = nn.Sequential(*conv_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(self.config.dropout)
        self.fc1 = nn.Linear(in_channels, self.config.hidden_dim)
        self.fc2 = nn.Linear(self.config.hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (batch, 1, n_features)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)  # (batch, channels)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
