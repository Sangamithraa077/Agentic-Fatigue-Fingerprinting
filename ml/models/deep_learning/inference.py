"""Deep learning inference and the sklearn-compatible adapter.

`TorchClassifierAdapter` exposes `.predict`/`.predict_proba` matching
scikit-learn's convention, so `ml.models.common.evaluation.evaluate_model`
can evaluate a trained `torch.nn.Module` through the exact same code path
used for classical models -- this is what "identical ... evaluation
protocols" means in practice for this benchmark.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn


class TorchClassifierAdapter:
    """Wraps a trained `torch.nn.Module` classifier for sklearn-style inference.

    Attributes:
        model: The trained module (moved to `device`, set to eval mode).
        device: Device inference runs on.
        batch_size: Mini-batch size used internally for large `X`.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None, batch_size: int = 256) -> None:
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device).eval()
        self.batch_size = batch_size

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, n_classes) softmax probabilities."""
        probabilities = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                batch = torch.as_tensor(X[start : start + self.batch_size], dtype=torch.float32, device=self.device)
                logits = self.model(batch)
                probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probabilities, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples,) predicted class indices (argmax of probabilities)."""
        return self.predict_proba(X).argmax(axis=1)
