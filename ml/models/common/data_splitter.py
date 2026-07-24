"""Group-aware data splitting utilities.

Both splitting strategies group by `SplitConfig.group_column` (default
`subject_id`) so that no group (subject) is split across train/test --
this specifically prevents leakage from Phase 8's overlapping (50%)
sliding windows, where naive row-level splitting could place near-duplicate
windows on both sides of a split. Grouping by subject also means these same
utilities can be reused unmodified for a future leave-one-subject-out
personalization phase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from ml.models.common.config import SplitConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    """One cross-validation fold's train/test row indices."""

    fold_index: int
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class TrainValTestSplit:
    """A single train/validation/test row-index split."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def stratified_group_kfold_split(y: np.ndarray, groups: np.ndarray, config: SplitConfig) -> List[Fold]:
    """Stratified group k-fold split, for classical model cross-validation.

    Preserves class balance across folds (stratified) while keeping every
    sample from one group (subject) entirely within one fold (group-aware).
    """
    n_groups = len(set(groups.tolist()))
    if n_groups < config.n_splits:
        raise ValueError(
            f"Cannot form {config.n_splits} group folds from only {n_groups} distinct group(s) "
            f"(group_column='{config.group_column}')."
        )

    splitter = StratifiedGroupKFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_seed)
    folds = [
        Fold(fold_index=i, train_idx=train_idx, test_idx=test_idx)
        for i, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y, groups))
    ]
    logger.info("Built %d stratified group folds across %d groups.", len(folds), n_groups)
    return folds


def train_val_test_group_split(y: np.ndarray, groups: np.ndarray, config: SplitConfig) -> TrainValTestSplit:
    """Group-aware train/validation/test split, for deep learning models.

    Splits in two steps (test first, then val from the remainder) so that
    the resulting proportions match `config.train_ratio`/`val_ratio`/
    `test_ratio` and no group appears in more than one of the three sets.
    """
    n_groups = len(set(groups.tolist()))
    if n_groups < 3:
        raise ValueError(
            f"Need at least 3 distinct groups for a train/val/test split, found {n_groups} "
            f"(group_column='{config.group_column}')."
        )

    indices = np.arange(len(y))

    test_splitter = GroupShuffleSplit(n_splits=1, test_size=config.test_ratio, random_state=config.random_seed)
    train_val_idx, test_idx = next(test_splitter.split(indices, y, groups))

    remaining_val_fraction = config.val_ratio / (config.train_ratio + config.val_ratio)
    val_splitter = GroupShuffleSplit(n_splits=1, test_size=remaining_val_fraction, random_state=config.random_seed)
    train_idx_rel, val_idx_rel = next(val_splitter.split(train_val_idx, y[train_val_idx], groups[train_val_idx]))

    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    logger.info(
        "Train/val/test group split: %d/%d/%d samples across %d groups.",
        len(train_idx), len(val_idx), len(test_idx), n_groups,
    )
    _assert_no_group_overlap(groups, train_idx, val_idx, test_idx)
    return TrainValTestSplit(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def _assert_no_group_overlap(groups: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> None:
    train_groups = set(groups[train_idx].tolist())
    val_groups = set(groups[val_idx].tolist())
    test_groups = set(groups[test_idx].tolist())
    overlap = (train_groups & val_groups) | (train_groups & test_groups) | (val_groups & test_groups)
    if overlap:
        raise RuntimeError(f"Group leakage detected across splits: {overlap}")
