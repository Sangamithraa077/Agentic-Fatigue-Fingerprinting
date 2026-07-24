"""CNN + BiLSTM hybrid baseline.

Purpose: the CNN front-end extracts local representations from the feature
vector (see `cnn1d.py`), and the BiLSTM back-end models sequence dynamics
over those extracted representations (see `bilstm.py`) -- combining both
architectures' inductive biases in one model, a common pattern in
sequence/signal classification literature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass(frozen=True)
class CNNBiLSTMConfig:
    """Architecture hyperparameters for `CNNBiLSTMClassifier`.

    Attributes:
        conv_channels: Output channels of the single CNN front-end block.
        kernel_size: Convolution kernel width.
        hidden_size: BiLSTM hidden state size (per direction).
        dropout: Dropout before the final linear layer.
    """

    conv_channels: int = 32
    kernel_size: int = 3
    hidden_size: int = 64
    dropout: float = 0.3


class CNNBiLSTMClassifier(nn.Module):
    """CNN feature extraction followed by BiLSTM sequence modeling.

    Input shape: `(batch, n_features)`. The CNN operates over
    `(batch, 1, n_features)`; its output channels then become the BiLSTM's
    per-step input size, with `n_features` as the sequence length.
    """

    def __init__(self, n_features: int, n_classes: int, config: Optional[CNNBiLSTMConfig] = None) -> None:
        super().__init__()
        self.config = config or CNNBiLSTMConfig()
        self.n_features = n_features
        self.n_classes = n_classes

        self.conv = nn.Sequential(
            nn.Conv1d(1, self.config.conv_channels, kernel_size=self.config.kernel_size, padding=self.config.kernel_size // 2),
            nn.BatchNorm1d(self.config.conv_channels),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=self.config.conv_channels,
            hidden_size=self.config.hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(self.config.dropout)
        self.fc = nn.Linear(self.config.hidden_size * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (batch, 1, n_features)
        x = self.conv(x)  # (batch, conv_channels, n_features)
        x = x.transpose(1, 2)  # (batch, n_features, conv_channels) -- seq_len=n_features
        _, (h_n, _) = self.lstm(x)
        combined = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(self.dropout(combined))
