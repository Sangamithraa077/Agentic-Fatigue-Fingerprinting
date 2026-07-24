"""Fusion module: combines the three per-modality embeddings into one
unified fatigue representation, via a configurable strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn, Tensor

from ml.models.proposed.attention import CrossModalAttention
from ml.models.proposed.config import FusionConfig


@dataclass
class FusionOutput:
    """Output of `FusionModule.forward`.

    Attributes:
        fused: The unified representation, shape (batch, fused_dim) -- see
            `FusionModule.fused_dim`.
        attention_weights: Per-layer attention weight tensors (see
            `attention.CrossModalAttentionOutput`), only populated when
            `strategy == "attention"`; `None` otherwise.
        modality_weights: The learned softmax-normalized per-modality
            scalar weights, only populated when `strategy == "weighted_sum"`;
            `None` otherwise.
        contextualized_embeddings: The per-modality embeddings *after*
            cross-modal attention (before pooling), shape
            (batch, num_modalities, embed_dim); only populated when
            `strategy == "attention"`. Exposed for later explainability
            reuse (per this phase's requirement to keep intermediate
            embeddings available).
    """

    fused: Tensor
    attention_weights: Optional[List[Tensor]] = None
    modality_weights: Optional[Tensor] = None
    contextualized_embeddings: Optional[Tensor] = None


class FusionModule(nn.Module):
    """Combines modality embeddings into one fatigue representation.

    `"concatenation"` tolerates each modality having a *different* embedding
    size (it just concatenates whatever it's given). `"attention"` and
    `"weighted_sum"` require every modality to share the same embedding
    size (stacking/weighting only make sense over equal-sized vectors) --
    validated here, at construction time, rather than by the caller, since
    this is exactly where the constraint originates.

    Example:
        >>> fusion = FusionModule(embed_dims={"emg": 32, "imu": 32, "ppg": 32}, modality_names=("emg","imu","ppg"), config=FusionConfig())
        >>> output = fusion({"emg": emg_embedding, "imu": imu_embedding, "ppg": ppg_embedding})
        >>> output.fused.shape
        torch.Size([batch, 32])
    """

    def __init__(self, embed_dims: Dict[str, int], modality_names: Sequence[str], config: FusionConfig) -> None:
        super().__init__()
        self.embed_dims = dict(embed_dims)
        self.modality_names = tuple(modality_names)
        self.config = config

        if config.strategy in ("attention", "weighted_sum"):
            unique_dims = {self.embed_dims[name] for name in self.modality_names}
            if len(unique_dims) > 1:
                raise ValueError(
                    f"Fusion strategy '{config.strategy}' requires every modality to share the same "
                    f"embedding size, got {self.embed_dims}. 'concatenation' is the only strategy "
                    "tolerating different sizes."
                )
            self.shared_embed_dim = next(iter(unique_dims))

        if config.strategy == "attention":
            if config.attention.embed_dim != self.shared_embed_dim:
                raise ValueError(
                    f"FusionConfig.attention.embed_dim ({config.attention.embed_dim}) must match "
                    f"the shared encoder output_dim ({self.shared_embed_dim})."
                )
            self.cross_attention = CrossModalAttention(config.attention)
        elif config.strategy == "weighted_sum":
            self.modality_logits = nn.Parameter(torch.zeros(len(self.modality_names)))

    @property
    def fused_dim(self) -> int:
        """The dimensionality of `forward(...).fused`."""
        if self.config.strategy == "concatenation":
            return sum(self.embed_dims[name] for name in self.modality_names)
        return self.shared_embed_dim

    def forward(self, embeddings: dict) -> FusionOutput:
        """Args: embeddings, a {modality_name: Tensor of shape (batch, embed_dim)} dict."""
        ordered = [embeddings[name] for name in self.modality_names]

        if self.config.strategy == "concatenation":
            return FusionOutput(fused=torch.cat(ordered, dim=1))

        stacked = torch.stack(ordered, dim=1)  # (batch, num_modalities, embed_dim)

        if self.config.strategy == "weighted_sum":
            weights = torch.softmax(self.modality_logits, dim=0)  # (num_modalities,)
            fused = (stacked * weights.view(1, -1, 1)).sum(dim=1)
            return FusionOutput(fused=fused, modality_weights=weights)

        # strategy == "attention"
        attention_output = self.cross_attention(stacked)
        fused = attention_output.embeddings.mean(dim=1)
        return FusionOutput(
            fused=fused,
            attention_weights=attention_output.attention_weights,
            contextualized_embeddings=attention_output.embeddings,
        )
