"""Classification head: maps the fused representation to class logits.

Supports both binary (`n_classes=2`) and multiclass output -- both are
handled identically here (a plain `n_classes`-wide linear output layer
trained with cross-entropy; see `losses.py`), since `CrossEntropyLoss`
generalizes correctly to the binary case without special-casing.
"""

from __future__ import annotations

from torch import nn, Tensor

from ml.models.proposed.config import ClassifierConfig


class ClassificationHead(nn.Module):
    """A configurable MLP classification head:
    `Linear -> [BatchNorm] -> ReLU -> Dropout`, repeated per
    `config.hidden_dims`, then a final `Linear` to `n_classes` logits (no
    softmax -- callers apply `torch.softmax` themselves, or rely on
    `nn.CrossEntropyLoss`, which expects raw logits).
    """

    def __init__(self, input_dim: int, n_classes: int, config: ClassifierConfig) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes

        layers = []
        in_features = input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, n_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Args: x, shape (batch, input_dim). Returns: logits, shape (batch, n_classes)."""
        return self.network(x)
