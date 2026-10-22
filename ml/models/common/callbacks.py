"""Training callbacks: early stopping and model checkpointing.

Framework-agnostic (pure Python, no torch import) so they are directly
unit testable and, in principle, reusable outside the deep-learning
training loop.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stops training when a monitored value stops improving.

    Example:
        >>> stopper = EarlyStopping(patience=10, mode="min")
        >>> for epoch_value in values:
        ...     if stopper.step(epoch_value):
        ...         break
    """

    def __init__(self, patience: int = 10, mode: str = "min", min_delta: float = 0.0) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self._best_value: Optional[float] = None
        self._epochs_without_improvement = 0
        self.best_epoch = -1
        self._epoch = -1

    @property
    def best_value(self) -> Optional[float]:
        return self._best_value

    def _is_improvement(self, value: float) -> bool:
        if self._best_value is None:
            return True
        if self.mode == "min":
            return value < self._best_value - self.min_delta
        return value > self._best_value + self.min_delta

    def step(self, value: float) -> bool:
        """Record one epoch's value. Returns True if training should stop."""
        self._epoch += 1
        if self._is_improvement(value):
            self._best_value = value
            self.best_epoch = self._epoch
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1

        should_stop = self._epochs_without_improvement >= self.patience
        if should_stop:
            logger.info(
                "Early stopping triggered at epoch %d (no improvement for %d epochs; best=%.6f at epoch %d).",
                self._epoch, self.patience, self._best_value, self.best_epoch,
            )
        return should_stop


class ModelCheckpoint:
    """Saves a model's state whenever the monitored value improves.

    `save_fn` is called as `save_fn(path)` only when `step` detects an
    improvement, decoupling this callback from any specific serialization
    mechanism (torch.save, joblib.dump, etc.).
    """

    def __init__(self, save_path: Path, mode: str = "min") -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.save_path = Path(save_path)
        self.mode = mode
        self._best_value: Optional[float] = None

    @property
    def best_value(self) -> Optional[float]:
        return self._best_value

    def step(self, value: float, save_fn) -> bool:
        """Record one epoch's value; call `save_fn(self.save_path)` and
        return True if `value` is the best seen so far."""
        is_improvement = self._best_value is None or (
            value < self._best_value if self.mode == "min" else value > self._best_value
        )
        if is_improvement:
            self._best_value = value
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            save_fn(self.save_path)
            logger.info("Checkpoint saved to '%s' (value=%.6f).", self.save_path, value)
        return is_improvement
