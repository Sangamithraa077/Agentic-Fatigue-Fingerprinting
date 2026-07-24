"""Inference and the sklearn-compatible adapter for the proposed model.

`ProposedModelAdapter` exposes `.predict`/`.predict_proba` over a single,
horizontally-concatenated feature array (EMG|IMU|PPG columns, matching
`ml.models.common.multimodal_dataset_builder.MultimodalDataset.X`) rather
than three separate arrays. This lets
`ml.models.common.evaluation.evaluate_model` (Phase 9's shared evaluation
path, reused here unmodified per this phase's requirement) treat this
multi-input model exactly like any single-input classical/deep-learning
model -- the adapter is the only place that knows about the 3-way split.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ml.models.common.multimodal_dataset_builder import ModalityDims
from ml.models.proposed.model import ModelOutput, ProposedFatigueModel


class ProposedModelAdapter:
    """Wraps a trained `ProposedFatigueModel` for sklearn-style inference
    over a concatenated (EMG|IMU|PPG) feature array.

    Attributes:
        model: The trained module (moved to `device`, set to eval mode).
        dims: Column counts used to split a concatenated array back into
            its three per-modality parts.
        device: Device inference runs on.
        batch_size: Mini-batch size used internally for large `X`.
    """

    def __init__(self, model: ProposedFatigueModel, dims: ModalityDims, device: Optional[torch.device] = None, batch_size: int = 256) -> None:
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device).eval()
        self.dims = dims
        self.batch_size = batch_size

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, n_classes) softmax probabilities."""
        probabilities = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                batch = X[start : start + self.batch_size]
                output = self._forward_batch(batch)
                probabilities.append(output.probabilities.cpu().numpy())
        return np.concatenate(probabilities, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples,) predicted class indices."""
        return self.predict_proba(X).argmax(axis=1)

    def predict_with_details(self, X: np.ndarray) -> ModelOutput:
        """Run inference on a single batch and return the full `ModelOutput`
        (attention weights, intermediate embeddings included) -- for later
        explainability/personalization reuse. Unlike `predict`/`predict_proba`,
        this does not internally chunk `X`; pass a batch that fits in memory.
        """
        with torch.no_grad():
            return self._forward_batch(X)

    def _forward_batch(self, X: np.ndarray) -> ModelOutput:
        x_emg, x_imu, x_ppg = self.dims.split(np.asarray(X, dtype=np.float32))
        return self.model(
            torch.as_tensor(x_emg, dtype=torch.float32, device=self.device),
            torch.as_tensor(x_imu, dtype=torch.float32, device=self.device),
            torch.as_tensor(x_ppg, dtype=torch.float32, device=self.device),
        )
