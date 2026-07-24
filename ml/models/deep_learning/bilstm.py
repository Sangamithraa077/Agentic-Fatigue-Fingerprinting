"""Bidirectional LSTM baseline.

Purpose: learn dependencies across the feature vector (treated as a
length-N sequence, one step per feature -- see `cnn1d.py` module docstring
for why) by scanning it in both directions and combining the final
forward/backward hidden states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass(frozen=True)
class BiLSTMConfig:
    """Architecture hyperparameters for `BiLSTMClassifier`.

    Attributes:
        hidden_size: LSTM hidden state size (per direction).
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout between stacked LSTM layers (ignored if
            `num_layers == 1`, matching `torch.nn.LSTM`'s own behavior)
            and before the final linear layer.
    """

    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.3


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM classifier over a feature vector.

    Input shape: `(batch, n_features)`. Internally reshaped to
    `(batch, n_features, 1)` so `nn.LSTM` treats each feature as one
    sequence step with a scalar input.
    """

    def __init__(self, n_features: int, n_classes: int, config: Optional[BiLSTMConfig] = None) -> None:
        super().__init__()
        self.config = config or BiLSTMConfig()
        self.n_features = n_features
        self.n_classes = n_classes

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(self.config.dropout)
        self.fc = nn.Linear(self.config.hidden_size * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)  # (batch, seq_len=n_features, input_size=1)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * 2, batch, hidden_size) -- last layer's forward/backward states
        forward_final = h_n[-2]
        backward_final = h_n[-1]
        combined = torch.cat([forward_final, backward_final], dim=1)
        return self.fc(self.dropout(combined))
