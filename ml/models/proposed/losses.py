"""Training loss construction for the proposed model.

Kept as its own module (rather than inlined in `training.py`) so the loss
-- including class-weight computation -- is independently testable and
reusable, e.g. by a future personalization phase re-weighting per subject.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn

from ml.models.proposed.config import LossConfig


def compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, normalized to mean 1.0.

    A class absent from `y` entirely gets weight 1.0 (not 0 or inf), so an
    unlucky train split missing a rare class doesn't corrupt the loss.
    """
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts[counts == 0] = len(y)  # avoid div-by-zero; effectively weight 1 after normalization below
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32)


def build_loss_function(config: Optional[LossConfig] = None, y_train: Optional[np.ndarray] = None, n_classes: Optional[int] = None) -> nn.Module:
    """Construct the configured training loss.

    Args:
        config: Loss configuration. Defaults to `LossConfig()`.
        y_train: Training labels, required if `config.use_class_weights`
            is True (the default).
        n_classes: Number of classes, required alongside `y_train`.

    Returns:
        An `nn.Module` loss callable as `loss(logits, targets)`.

    Raises:
        ValueError: If `config.name` is not `"cross_entropy"`, or class
            weights are requested but `y_train`/`n_classes` are missing.
    """
    config = config or LossConfig()

    if config.name != "cross_entropy":
        raise ValueError(f"Unknown loss name: {config.name!r}")

    class_weights = None
    if config.use_class_weights:
        if y_train is None or n_classes is None:
            raise ValueError("use_class_weights=True requires both y_train and n_classes.")
        class_weights = compute_class_weights(y_train, n_classes)

    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.label_smoothing)
