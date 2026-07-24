"""Configuration dataclasses for the proposed multimodal fusion model.

Mirrors `ml.models.common.config`'s philosophy: every experiment is fully
described by a combination of these config objects, so a run is
reproducible from its configuration alone. Nothing here builds a model or
touches data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class EncoderConfig:
    """Configuration for one modality's MLP encoder (see `encoders.py`).

    Attributes:
        hidden_dims: Sizes of the hidden layers, in order. The final layer
            always projects to `output_dim` (not included here).
        output_dim: The fixed-size embedding dimension every modality
            encoder produces -- must match across all three encoders so
            they can be fused (concatenation is the one exception, which
            only needs a consistent per-modality size, not cross-modality
            equality, but a single shared `output_dim` keeps the model
            simple and is required by the attention and weighted-sum
            fusion strategies).
        dropout: Dropout probability after each hidden layer's activation.
        use_batch_norm: Whether to apply `BatchNorm1d` after each hidden
            linear layer (before the activation).
    """

    hidden_dims: Tuple[int, ...] = (128, 64)
    output_dim: int = 32
    dropout: float = 0.3
    use_batch_norm: bool = True


@dataclass(frozen=True)
class AttentionConfig:
    """Configuration for the multi-head cross-modal attention stack (see
    `attention.py`).

    Attributes:
        embed_dim: Must equal each `EncoderConfig.output_dim` (the
            attention operates directly on the per-modality embeddings).
        num_heads: Number of attention heads (`embed_dim` must be evenly
            divisible by this).
        dropout: Dropout applied inside `nn.MultiheadAttention` and on the
            residual connection.
        num_layers: Number of stacked cross-modal attention layers.
    """

    embed_dim: int = 32
    num_heads: int = 4
    dropout: float = 0.1
    num_layers: int = 1

    def __post_init__(self) -> None:
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads}).")


@dataclass(frozen=True)
class FusionConfig:
    """Configuration for the fusion module (see `fusion.py`).

    Attributes:
        strategy: `"attention"` (default -- multi-head cross-modal
            attention followed by mean-pooling across modalities),
            `"concatenation"` (simple concatenation of the per-modality
            embeddings, no learned cross-modal interaction), or
            `"weighted_sum"` (a learned softmax-normalized scalar weight
            per modality).
        attention: Only used when `strategy == "attention"`.
    """

    strategy: str = "attention"
    attention: AttentionConfig = field(default_factory=AttentionConfig)

    def __post_init__(self) -> None:
        if self.strategy not in ("concatenation", "attention", "weighted_sum"):
            raise ValueError(f"Unknown fusion strategy: {self.strategy!r}")


@dataclass(frozen=True)
class ClassifierConfig:
    """Configuration for the classification head (see `classifier.py`).

    Attributes:
        hidden_dims: Sizes of the hidden layers before the final
            `n_classes`-wide output layer.
        dropout: Dropout probability after each hidden layer's activation.
        use_batch_norm: Whether to apply `BatchNorm1d` after each hidden
            linear layer.
    """

    hidden_dims: Tuple[int, ...] = (64,)
    dropout: float = 0.3
    use_batch_norm: bool = True


@dataclass(frozen=True)
class LossConfig:
    """Configuration for the training loss (see `losses.py`).

    Attributes:
        name: Currently only `"cross_entropy"` is implemented.
        label_smoothing: Standard label-smoothing regularization.
        use_class_weights: If True, the loss is weighted inversely to
            class frequency (computed from the training labels at
            training time) -- appropriate given this project's documented
            class imbalance.
    """

    name: str = "cross_entropy"
    label_smoothing: float = 0.0
    use_class_weights: bool = True


@dataclass(frozen=True)
class ProposedModelConfig:
    """Top-level architecture configuration bundling every sub-config.

    Attributes:
        emg_encoder, imu_encoder, ppg_encoder: Per-modality encoder configs.
            Each `output_dim` must match for `"attention"`/`"weighted_sum"`
            fusion (validated in `model.py`).
        fusion: Fusion strategy configuration.
        classifier: Classification head configuration.
        n_classes: Number of output classes (2 for binary, >2 for multiclass).
    """

    emg_encoder: EncoderConfig = field(default_factory=EncoderConfig)
    imu_encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(hidden_dims=(256, 64)))
    ppg_encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(hidden_dims=(16,)))
    fusion: FusionConfig = field(default_factory=FusionConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    n_classes: int = 2


@dataclass(frozen=True)
class ProposedTrainingConfig:
    """Training loop configuration for the proposed model.

    Mirrors `ml.models.common.config.TrainingConfig` (Phase 9), with one
    addition (`use_mixed_precision`) specific to this larger model.

    Attributes:
        batch_size, max_epochs, learning_rate, weight_decay: Standard
            mini-batch training hyperparameters.
        early_stopping_patience, early_stopping_min_delta: Passed to
            `ml.models.common.callbacks.EarlyStopping`.
        lr_scheduler_patience, lr_scheduler_factor: Passed to
            `torch.optim.lr_scheduler.ReduceLROnPlateau`.
        random_seed: Seed for `random`/`numpy`/`torch` reproducibility.
        device: `"auto"` (CUDA if available, else CPU), `"cpu"`, or `"cuda"`.
        use_mixed_precision: If True *and* running on a CUDA device,
            training uses `torch.autocast` + `GradScaler`. Silently has no
            effect on CPU (mixed precision only benefits CUDA tensor cores).
        checkpoint_path: Where to save the best-validation-loss checkpoint.
            `None` disables checkpointing.
    """

    batch_size: int = 64
    max_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    random_seed: int = 42
    device: str = "auto"
    use_mixed_precision: bool = True
    checkpoint_path: Optional[str] = None
