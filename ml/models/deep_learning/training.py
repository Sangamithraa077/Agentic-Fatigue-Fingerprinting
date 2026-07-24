"""Reusable deep learning training loop.

Shared by every architecture in `ml/models/deep_learning/` (CNN1D, BiLSTM,
CNN-BiLSTM, TCN) and, per this phase's closing requirement, intended to be
reusable unmodified by the later multimodal-attention/personalization
phases -- nothing here is architecture-specific.

Supports: mini-batch training, early stopping, LR scheduling on plateau,
best-epoch checkpointing, seeded reproducibility, and automatic GPU use
when available.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from ml.models.common.callbacks import EarlyStopping, ModelCheckpoint
from ml.models.common.config import TrainingConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingHistory:
    """Per-epoch training record, for learning/loss curve plots."""

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    epochs_trained: int = 0
    best_epoch: int = -1
    training_seconds: float = 0.0


def set_seed(seed: int) -> None:
    """Seed `random`, `numpy`, and `torch` (CPU and CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> torch.device:
    """Resolve a device preference ("auto"/"cpu"/"cuda") to a `torch.device`,
    falling back to CPU if CUDA was requested/preferred but unavailable."""
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available -- using CPU instead.")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _run_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device, train: bool) -> tuple:
    model.train(mode=train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            if train:
                loss.backward()
                optimizer.step()

            batch_size = y_batch.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total_count += batch_size

    return total_loss / total_count, total_correct / total_count


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainingConfig,
    checkpoint_path: Optional[Path] = None,
) -> TrainingHistory:
    """Train `model` with mini-batch SGD (Adam), early stopping, LR-on-plateau
    scheduling, and (optionally) best-epoch checkpointing.

    Args:
        model: A `torch.nn.Module` classifier (logits output, no softmax).
        X_train, y_train, X_val, y_val: Already-scaled/imputed arrays (see
            `ml.models.common.dataset_builder`).
        config: Training hyperparameters (see `TrainingConfig`).
        checkpoint_path: If given, the best-validation-loss epoch's model
            weights are saved here via `torch.save`.

    Returns:
        A `TrainingHistory` with per-epoch loss/accuracy curves.

    Mutates `model` in place: after training, `model`'s weights are the
    best-validation-loss epoch's weights (restored from the in-memory best
    state), not necessarily the final epoch's weights.
    """
    set_seed(config.random_seed)
    device = get_device(config.device)
    model.to(device)

    train_loader = _make_loader(X_train, y_train, config.batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val, config.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=config.lr_scheduler_factor, patience=config.lr_scheduler_patience)
    criterion = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(patience=config.early_stopping_patience, mode="min", min_delta=config.early_stopping_min_delta)
    checkpoint = ModelCheckpoint(checkpoint_path, mode="min") if checkpoint_path is not None else None

    history = TrainingHistory()
    best_state: Optional[dict] = None
    start_time = time.perf_counter()

    for epoch in range(config.max_epochs):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_accuracy.append(train_acc)
        history.val_accuracy.append(val_acc)

        if early_stopping.best_value is None or val_loss < early_stopping.best_value - config.early_stopping_min_delta:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if checkpoint is not None:
            checkpoint.step(val_loss, lambda path: torch.save(model.state_dict(), path))

        logger.info(
            "Epoch %d/%d: train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch + 1, config.max_epochs, train_loss, train_acc, val_loss, val_acc,
        )

        if early_stopping.step(val_loss):
            break

    history_dict_epochs = len(history.train_loss)
    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainingHistory(
        train_loss=history.train_loss,
        val_loss=history.val_loss,
        train_accuracy=history.train_accuracy,
        val_accuracy=history.val_accuracy,
        epochs_trained=history_dict_epochs,
        best_epoch=early_stopping.best_epoch,
        training_seconds=time.perf_counter() - start_time,
    )
