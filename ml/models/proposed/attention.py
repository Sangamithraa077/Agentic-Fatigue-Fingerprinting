"""Multi-head cross-modal attention.

Treats the 3 modality embeddings (EMG, IMU, PPG -- each already projected
to the same `embed_dim` by their respective `ModalityEncoder`) as a
length-3 sequence of "tokens" and applies `torch.nn.MultiheadAttention` as
self-attention over that sequence. This lets each modality's
representation be updated based on (attend to) the other modalities --
the "cross-modal" interaction -- while reusing PyTorch's standard,
well-tested attention implementation rather than a bespoke one.

Attention weights are returned alongside the contextualized embeddings
specifically so a later phase can inspect *which* modality most influenced
a given prediction (per the Phase 10 instruction to expose them for reuse,
ahead of SHAP-based explainability).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from torch import nn, Tensor

from ml.models.proposed.config import AttentionConfig


@dataclass
class CrossModalAttentionOutput:
    """Output of the full cross-modal attention stack.

    Attributes:
        embeddings: Contextualized per-modality embeddings, shape
            (batch, num_modalities, embed_dim) -- same shape as the input,
            updated by attending across modalities.
        attention_weights: One tensor per attention layer, each shape
            (batch, num_heads, num_modalities, num_modalities) (row i,
            column j = how much modality i attends to modality j, per
            head). Kept per-head (not averaged) and per-layer, so a later
            explainability phase has the finest-grained view available.
    """

    embeddings: Tensor
    attention_weights: List[Tensor]


class CrossModalAttentionLayer(nn.Module):
    """One residual cross-modal attention block: multi-head self-attention
    over the modality-token sequence, followed by a residual connection and
    layer normalization (the standard transformer-encoder attention
    sub-block)."""

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=config.embed_dim, num_heads=config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Args: x, shape (batch, num_modalities, embed_dim).

        Returns:
            (contextualized_x, attention_weights) -- attention_weights has
            shape (batch, num_heads, num_modalities, num_modalities).
        """
        attn_output, attn_weights = self.attention(x, x, x, need_weights=True, average_attn_weights=False)
        return self.norm(x + self.dropout(attn_output)), attn_weights


class CrossModalAttention(nn.Module):
    """A configurable stack of `CrossModalAttentionLayer`s."""

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([CrossModalAttentionLayer(config) for _ in range(config.num_layers)])

    def forward(self, x: Tensor) -> CrossModalAttentionOutput:
        """Args: x, shape (batch, num_modalities, embed_dim)."""
        all_weights: List[Tensor] = []
        for layer in self.layers:
            x, weights = layer(x)
            all_weights.append(weights)
        return CrossModalAttentionOutput(embeddings=x, attention_weights=all_weights)
