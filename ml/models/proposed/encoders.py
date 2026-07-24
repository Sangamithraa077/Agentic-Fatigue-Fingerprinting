"""Per-modality MLP encoders.

Each of EMG/IMU/PPG gets its own independent encoder, mapping that
modality's raw (and very differently sized -- EMG=72, IMU=558, PPG=7
features in this project's default multimodal dataset) feature vector down
to a common fixed-size embedding, so the fusion module downstream never
needs to know each modality's original dimensionality.
"""

from __future__ import annotations

from torch import nn, Tensor

from ml.models.proposed.config import EncoderConfig


class ModalityEncoder(nn.Module):
    """A configurable MLP: `Linear -> [BatchNorm] -> ReLU -> Dropout`,
    repeated per `config.hidden_dims`, then a final `Linear` to
    `config.output_dim` (no activation on the output -- it feeds directly
    into fusion/attention, not a probability).
    """

    def __init__(self, input_dim: int, config: EncoderConfig) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = config.output_dim

        layers = []
        in_features = input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, config.output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Args: x, shape (batch, input_dim). Returns: shape (batch, output_dim)."""
        return self.network(x)
