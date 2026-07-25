"""Explainability views over the proposed model's cross-modal attention
weights (`ml.models.proposed.attention.CrossModalAttentionOutput.attention_weights`,
only populated when `FusionConfig.strategy == "attention"`).

These are **raw-tensor** explanations -- summarizing and visualizing
attention weights the model already computes -- not SHAP. Attention
weights are a useful, cheap complement to SHAP (they show what the *model*
attends to internally) but are not a substitute for it (attention weights
are not a rigorous attribution method; see `docs/`/the Phase 11 report's
Limitations section).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import Tensor

from ml.models.proposed.model import MODALITY_NAMES


@dataclass(frozen=True)
class AttentionMap:
    """One attention layer's modality-by-modality weight matrix.

    Attributes:
        layer_index: Which stacked `CrossModalAttentionLayer` this is from.
        matrix: (n_modalities, n_modalities) attention weights, averaged
            over the batch and attention heads; row i, column j = how much
            modality i attends to modality j.
        modality_names: Row/column labels, in `matrix`'s index order.
    """

    layer_index: int
    matrix: np.ndarray
    modality_names: Tuple[str, ...]


@dataclass(frozen=True)
class ModalityImportance:
    """One modality's overall importance to the fused representation.

    Attributes:
        modality_name: The modality.
        importance: Normalized (sums to 1 across all modalities) average
            attention *received* by this modality -- i.e. how much every
            other modality (and itself), across every layer/head/sample,
            relies on this modality's representation.
    """

    modality_name: str
    importance: float


@dataclass(frozen=True)
class AttentionDistribution:
    """Summary statistics of one (layer, head)'s attention weight distribution."""

    layer_index: int
    head_index: int
    mean: float
    std: float
    min: float
    max: float


def generate_attention_map(
    attention_weights: Sequence[Tensor], modality_names: Sequence[str] = MODALITY_NAMES
) -> Tuple[AttentionMap, ...]:
    """Reduce each layer's `(batch, num_heads, n_modalities, n_modalities)`
    attention weights to one `(n_modalities, n_modalities)` map, averaged
    over the batch and heads (the finest-grained per-head/per-sample
    tensors remain available on the model's raw `ModelOutput` for any
    caller that needs them -- this is the aggregate view for reporting)."""
    maps: List[AttentionMap] = []
    for layer_index, layer_weights in enumerate(attention_weights):
        averaged = layer_weights.mean(dim=(0, 1)).detach().cpu().numpy()
        maps.append(AttentionMap(layer_index=layer_index, matrix=averaged, modality_names=tuple(modality_names)))
    return tuple(maps)


def compute_modality_importance(
    attention_weights: Sequence[Tensor], modality_names: Sequence[str] = MODALITY_NAMES
) -> Tuple[ModalityImportance, ...]:
    """Average, over every layer/head/sample/querying-modality, how much
    attention each modality receives -- a single importance score per
    modality summarizing `generate_attention_map`'s per-layer detail."""
    stacked = torch.stack(list(attention_weights), dim=0)  # (n_layers, batch, heads, query, key)
    received = stacked.mean(dim=(0, 1, 2, 3)).detach().cpu().numpy()  # (n_modalities,) -- averaged over the key axis's complement
    total = received.sum()
    normalized = received / total if total > 0 else received

    return tuple(
        ModalityImportance(modality_name=name, importance=float(value)) for name, value in zip(modality_names, normalized)
    )


def compute_attention_distributions(attention_weights: Sequence[Tensor]) -> Tuple[AttentionDistribution, ...]:
    """Per-(layer, head) summary statistics of the attention weight distribution."""
    distributions: List[AttentionDistribution] = []
    for layer_index, layer_weights in enumerate(attention_weights):
        weights = layer_weights.detach().cpu().numpy()  # (batch, heads, query, key)
        for head_index in range(weights.shape[1]):
            head_values = weights[:, head_index, :, :].reshape(-1)
            distributions.append(
                AttentionDistribution(
                    layer_index=layer_index, head_index=head_index,
                    mean=float(head_values.mean()), std=float(head_values.std()),
                    min=float(head_values.min()), max=float(head_values.max()),
                )
            )
    return tuple(distributions)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _save_and_close(fig: plt.Figure, save_path: Path) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_attention_heatmap(attention_map: AttentionMap, save_path: Path) -> Path:
    """Heatmap of one layer's querying-modality x attended-to-modality weights."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(
        attention_map.matrix, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1,
        xticklabels=attention_map.modality_names, yticklabels=attention_map.modality_names, ax=ax,
    )
    ax.set_xlabel("Attended-to modality")
    ax.set_ylabel("Querying modality")
    ax.set_title(f"Cross-Modal Attention -- Layer {attention_map.layer_index}")
    fig.tight_layout()
    return _save_and_close(fig, save_path)


def plot_modality_importance(importances: Sequence[ModalityImportance], save_path: Path) -> Path:
    """Bar chart of each modality's normalized overall importance."""
    fig, ax = plt.subplots(figsize=(6, 4))
    names = [item.modality_name for item in importances]
    values = [item.importance for item in importances]
    ax.bar(names, values, color=sns.color_palette("viridis", len(names)))
    ax.set_ylabel("Relative importance (normalized attention received)")
    ax.set_title("Modality Importance (Cross-Modal Attention)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return _save_and_close(fig, save_path)


def plot_attention_distribution(distributions: Sequence[AttentionDistribution], save_path: Path) -> Path:
    """Bar chart (mean +/- std) of every (layer, head)'s attention weight distribution."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [f"L{d.layer_index}H{d.head_index}" for d in distributions]
    means = [d.mean for d in distributions]
    stds = [d.std for d in distributions]
    ax.bar(labels, means, yerr=stds, capsize=4, color="#4C72B0")
    ax.set_ylabel("Attention weight (mean +/- std)")
    ax.set_title("Attention Weight Distribution by Layer/Head")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return _save_and_close(fig, save_path)
