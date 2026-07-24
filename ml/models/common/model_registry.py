"""Central catalog of every available baseline model, with availability
tracking (e.g. XGBoost's graceful fallback) so callers can discover what's
usable without importing every model module directly.

This is a lookup/metadata registry rather than a uniform factory, because
classical and deep learning models have genuinely different construction
signatures (classical: `factory(config)`; deep learning: `factory(n_features,
n_classes, config)`, since the input/output dimensionality isn't known until
the dataset is built) -- forcing one signature would be a false abstraction.
Each model's own module (e.g. `classical/random_forest.py`) exposes its
`build_*` factory directly; this registry's job is discovery and
availability, which *is* uniform across both categories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelSpec:
    """Metadata describing one registered model."""

    name: str
    category: str  # "classical" | "deep_learning"
    description: str
    is_available: bool = True
    unavailable_reason: Optional[str] = None


class ModelRegistry:
    """A simple name -> `ModelSpec` catalog."""

    def __init__(self) -> None:
        self._specs: Dict[str, ModelSpec] = {}

    def register(self, spec: ModelSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Model '{spec.name}' is already registered.")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ModelSpec:
        if name not in self._specs:
            raise KeyError(f"Model '{name}' is not registered. Known models: {sorted(self._specs)}")
        return self._specs[name]

    def list_names(self, category: Optional[str] = None, available_only: bool = False) -> List[str]:
        specs = self._specs.values()
        if category is not None:
            specs = (s for s in specs if s.category == category)
        if available_only:
            specs = (s for s in specs if s.is_available)
        return sorted(s.name for s in specs)


def build_default_registry() -> ModelRegistry:
    """Build the registry covering every model implemented in this phase."""
    from ml.models.classical.xgboost import check_xgboost_availability

    registry = ModelRegistry()

    registry.register(ModelSpec("logistic_regression", "classical", "Linear baseline, L2-regularized."))
    registry.register(ModelSpec("svm", "classical", "Kernel support vector machine (RBF)."))
    registry.register(ModelSpec("random_forest", "classical", "Bagged ensemble of decision trees."))

    xgboost_available, unavailable_reason = check_xgboost_availability()
    registry.register(
        ModelSpec(
            "xgboost", "classical", "Gradient-boosted trees.",
            is_available=xgboost_available, unavailable_reason=unavailable_reason,
        )
    )

    registry.register(ModelSpec("cnn1d", "deep_learning", "1D CNN over the feature vector -- learns local patterns."))
    registry.register(ModelSpec("bilstm", "deep_learning", "Bidirectional LSTM over the feature vector -- learns sequence dependencies."))
    registry.register(
        ModelSpec("cnn_bilstm", "deep_learning", "CNN local feature extraction followed by BiLSTM sequence modeling.")
    )
    registry.register(
        ModelSpec("tcn", "deep_learning", "Temporal Convolutional Network -- dilated causal convolutions for long-range context.")
    )

    return registry
