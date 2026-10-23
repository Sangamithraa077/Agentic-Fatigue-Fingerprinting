"""Training loop for the proposed multimodal model.

Reuses Phase 9's common training framework directly rather than
duplicating it: `EarlyStopping`/`ModelCheckpoint` from
`ml.models.common.callbacks`, and `set_seed`/`get_device` from
`ml.models.deep_learning.training` (both are input-shape-agnostic utility
functions, not tied to Phase 9's single-input models).

The mini-batch loop itself is a new implementation rather than a call into
Phase 9's `ml.models.deep_learning.training.train_model`, because that
function's signature takes one input tensor per sample -- this model takes
three (EMG, IMU, PPG). Reusing the *shared primitives* while writing a
thin, input-shape-appropriate loop is the reuse boundary this phase's
"no duplicated logic" principle actually calls for: the duplicated-looking
loop body is an unavoidable, minimal consequence of a genuinely different
input arity, not copy-pasted business logic.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from ml.models.common.callbacks import EarlyStopping, ModelCheckpoint
from ml.models.deep_learning.training import get_device, set_seed
from ml.models.proposed.config import LossConfig, ProposedTrainingConfig
from ml.models.proposed.losses import build_loss_function
from ml.models.proposed.model import ProposedFatigueModel
from ml.preprocessing.augmentation import TrainingDataAugmenter

logger = logging.getLogger(__name__)


@dataclass
class ProposedTrainingHistory:
    """Per-epoch training record, for learning/loss curve plots."""

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    epochs_trained: int = 0
    best_epoch: int = -1
    training_seconds: float = 0.0


def _make_loader(x_emg: np.ndarray, x_imu: np.ndarray, x_ppg: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(x_emg, dtype=torch.float32),
        torch.as_tensor(x_imu, dtype=torch.float32),
        torch.as_tensor(x_ppg, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _run_epoch(
    model: ProposedFatigueModel,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    train: bool,
    use_mixed_precision: bool,
    scaler: Optional["torch.amp.GradScaler"],
) -> tuple:
    model.train(mode=train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    autocast_enabled = use_mixed_precision and device.type == "cuda"

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x_emg, x_imu, x_ppg, y in loader:
            x_emg, x_imu, x_ppg, y = x_emg.to(device), x_imu.to(device), x_ppg.to(device), y.to(device)
            if train:
                optimizer.zero_grad()

            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                output = model(x_emg, x_imu, x_ppg)
                loss = criterion(output.logits, y)

            if train:
                if scaler is not None and autocast_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (output.predictions == y).sum().item()
            total_count += batch_size

    return total_loss / total_count, total_correct / total_count


def train_proposed_model(
    model: ProposedFatigueModel,
    x_emg_train: np.ndarray,
    x_imu_train: np.ndarray,
    x_ppg_train: np.ndarray,
    y_train: np.ndarray,
    x_emg_val: np.ndarray,
    x_imu_val: np.ndarray,
    x_ppg_val: np.ndarray,
    y_val: np.ndarray,
    training_config: ProposedTrainingConfig,
    loss_config: Optional[LossConfig] = None,
    checkpoint_path: Optional[Path] = None,
    augment_training: bool = False,
) -> ProposedTrainingHistory:
    """Train `model` with mini-batch Adam, early stopping, LR-on-plateau
    scheduling, optional mixed precision (CUDA only), and best-epoch
    checkpointing.

    Applies Gaussian-noise + amplitude-scaling augmentation to the
    training split (train-only; validation data is never augmented) when
    `augment_training=True`. Each of the three modality arrays is augmented
    independently with the same seed so that the augmented rows still
    correspond to the same label in `y_train`.

    Off by default: the committed benchmark reports were produced without
    augmentation, and the leave-one-participant-out study in `research/` found
    that noise injected into extracted feature tables worsened cross-athlete
    generalization. Enable it only as an experiment you evaluate.

    Mutates `model` in place: after training, `model`'s weights are the
    best-validation-loss epoch's weights, not necessarily the final
    epoch's.
    """
    set_seed(training_config.random_seed)
    device = get_device(training_config.device)
    model.to(device)

    # ------------------------------------------------------------------
    # Augment training data ONLY (validation/test data is never touched)
    # ------------------------------------------------------------------
    if augment_training:
        augmenter = TrainingDataAugmenter(
            use_noise=True,
            use_scaling=True,
            use_warping=False,   # warping disabled by default (can flip for experiments)
            noise_ratio=0.02,
            scale_range=(0.95, 1.05),
            seed=training_config.random_seed,
        )
        logger.info("Applying training-data augmentation (noise + scaling, multiplier=1) ...")
        x_emg_train, y_aug     = augmenter.augment(x_emg_train, y_train, multiplier=1)
        x_imu_train, _         = augmenter.augment(x_imu_train, y_train, multiplier=1)
        x_ppg_train, _         = augmenter.augment(x_ppg_train, y_train, multiplier=1)
        y_train                 = y_aug
        logger.info("Augmented training set size: %d samples (was %d).", len(y_train), len(y_train) // 2)
    else:
        logger.info("Training-data augmentation skipped (augment_training=False).")

    train_loader = _make_loader(x_emg_train, x_imu_train, x_ppg_train, y_train, training_config.batch_size, shuffle=True)
    val_loader = _make_loader(x_emg_val, x_imu_val, x_ppg_val, y_val, training_config.batch_size, shuffle=False)

    criterion = build_loss_function(loss_config, y_train=y_train, n_classes=model.config.n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=training_config.lr_scheduler_factor, patience=training_config.lr_scheduler_patience)

    use_mixed_precision = training_config.use_mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision) if device.type == "cuda" else None
    if training_config.use_mixed_precision and device.type != "cuda":
        logger.info("Mixed precision requested but device is '%s' -- has no effect off CUDA.", device.type)

    early_stopping = EarlyStopping(
        patience=training_config.early_stopping_patience, mode="min", min_delta=training_config.early_stopping_min_delta
    )
    checkpoint = ModelCheckpoint(Path(checkpoint_path), mode="min") if checkpoint_path is not None else None

    history = ProposedTrainingHistory()
    best_state: Optional[dict] = None
    start_time = time.perf_counter()

    for epoch in range(training_config.max_epochs):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device, True, use_mixed_precision, scaler)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer, device, False, use_mixed_precision, scaler)
        scheduler.step(val_loss)

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_accuracy.append(train_acc)
        history.val_accuracy.append(val_acc)

        if early_stopping.best_value is None or val_loss < early_stopping.best_value - training_config.early_stopping_min_delta:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if checkpoint is not None:
            checkpoint.step(val_loss, lambda path: torch.save(model.state_dict(), path))

        logger.info(
            "Epoch %d/%d: train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch + 1, training_config.max_epochs, train_loss, train_acc, val_loss, val_acc,
        )

        if early_stopping.step(val_loss):
            break

    epochs_trained = len(history.train_loss)
    if best_state is not None:
        model.load_state_dict(best_state)

    return ProposedTrainingHistory(
        train_loss=history.train_loss,
        val_loss=history.val_loss,
        train_accuracy=history.train_accuracy,
        val_accuracy=history.val_accuracy,
        epochs_trained=epochs_trained,
        best_epoch=early_stopping.best_epoch,
        training_seconds=time.perf_counter() - start_time,
    )
