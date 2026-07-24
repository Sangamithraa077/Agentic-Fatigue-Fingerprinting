"""The proposed multimodal fusion model: independent EMG/IMU/PPG encoders,
attention-based (or concatenation/weighted-sum) fusion, and a
classification head -- this project's novel architectural contribution.

Explicitly out of scope for this phase (later phases): SHAP explainability,
subject personalization, agentic reasoning. This module only exposes the
raw ingredients (attention weights, intermediate embeddings) those phases
will need.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from torch import nn, Tensor

from ml.models.proposed.classifier import ClassificationHead
from ml.models.proposed.config import ProposedModelConfig
from ml.models.proposed.encoders import ModalityEncoder
from ml.models.proposed.fusion import FusionModule

MODALITY_NAMES = ("emg", "imu", "ppg")


@dataclass
class ModelOutput:
    """Complete output of one forward pass, per this phase's requirement to
    return predictions, probabilities, attention weights, and intermediate
    embeddings (not just logits).

    Attributes:
        logits: Raw classifier output, shape (batch, n_classes).
        probabilities: `softmax(logits)`, shape (batch, n_classes).
        predictions: `argmax(probabilities)`, shape (batch,).
        modality_embeddings: Each modality's independently-encoded
            embedding (before fusion), shape (batch, embed_dim) each --
            the "intermediate embeddings" this phase is required to expose
            for later explainability/personalization reuse.
        fused_embedding: The single unified representation fed to the
            classifier, shape (batch, fused_dim).
        attention_weights: Per-layer cross-modal attention weights (only
            populated when the fusion strategy is `"attention"`); see
            `attention.CrossModalAttentionOutput`.
        contextualized_embeddings: Per-modality embeddings after
            cross-modal attention, before pooling (only populated when the
            fusion strategy is `"attention"`), shape
            (batch, num_modalities, embed_dim).
        modality_weights: Learned per-modality scalar weights (only
            populated when the fusion strategy is `"weighted_sum"`).
    """

    logits: Tensor
    probabilities: Tensor
    predictions: Tensor
    modality_embeddings: Dict[str, Tensor]
    fused_embedding: Tensor
    attention_weights: Optional[List[Tensor]] = None
    contextualized_embeddings: Optional[Tensor] = None
    modality_weights: Optional[Tensor] = None


class ProposedFatigueModel(nn.Module):
    """Independent modality encoders -> configurable fusion -> classifier.

    Example:
        >>> config = ProposedModelConfig(n_classes=2)
        >>> model = ProposedFatigueModel(emg_dim=72, imu_dim=558, ppg_dim=7, config=config)
        >>> output = model(x_emg, x_imu, x_ppg)
        >>> output.predictions.shape
        torch.Size([batch])
    """

    def __init__(self, emg_dim: int, imu_dim: int, ppg_dim: int, config: ProposedModelConfig) -> None:
        super().__init__()
        self.config = config

        self.encoders = nn.ModuleDict(
            {
                "emg": ModalityEncoder(emg_dim, config.emg_encoder),
                "imu": ModalityEncoder(imu_dim, config.imu_encoder),
                "ppg": ModalityEncoder(ppg_dim, config.ppg_encoder),
            }
        )

        embed_dims = {name: encoder.output_dim for name, encoder in self.encoders.items()}
        self.fusion = FusionModule(embed_dims=embed_dims, modality_names=MODALITY_NAMES, config=config.fusion)
        self.classifier = ClassificationHead(input_dim=self.fusion.fused_dim, n_classes=config.n_classes, config=config.classifier)

    def forward(self, x_emg: Tensor, x_imu: Tensor, x_ppg: Tensor) -> ModelOutput:
        """Args: x_emg/x_imu/x_ppg, each shape (batch, modality_feature_dim)."""
        modality_embeddings = {
            "emg": self.encoders["emg"](x_emg),
            "imu": self.encoders["imu"](x_imu),
            "ppg": self.encoders["ppg"](x_ppg),
        }

        fusion_output = self.fusion(modality_embeddings)
        logits = self.classifier(fusion_output.fused)
        probabilities = logits.softmax(dim=1)
        predictions = probabilities.argmax(dim=1)

        return ModelOutput(
            logits=logits,
            probabilities=probabilities,
            predictions=predictions,
            modality_embeddings=modality_embeddings,
            fused_embedding=fusion_output.fused,
            attention_weights=fusion_output.attention_weights,
            contextualized_embeddings=fusion_output.contextualized_embeddings,
            modality_weights=fusion_output.modality_weights,
        )
