"""Subject similarity search over `subject_profiles.SubjectProfile.summary_vector`.

Every profile's `summary_vector` mixes scales that are not directly
comparable (age in years, mass in kilograms, raw EMG feature means near
zero, ...), so this module standardizes (zero mean, unit variance) across
the given population of profiles *before* computing similarity -- a
similarity.py-level concern, since only the population being searched over
(not any single profile in isolation) determines the right standardization
statistics.

Both supported metrics are normalized to a common "higher = more similar"
convention so callers never need to know which metric produced a given
`SimilarityResult.score`: cosine similarity is used as-is (already
higher-is-more-similar, range roughly `[-1, 1]`); Euclidean distance is
converted via `1 / (1 + distance)` (range `(0, 1]`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from ml.personalization.config import SimilarityConfig
from ml.personalization.subject_profiles import SubjectProfile


@dataclass(frozen=True)
class SimilarityResult:
    """One candidate subject's similarity to a query subject.

    Attributes:
        subject_id: The candidate subject.
        score: Similarity score -- higher means more similar, regardless of
            which metric produced it (see module docstring).
    """

    subject_id: str
    score: float


def standardize_rows(matrix: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance standardize each column of `matrix`."""
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    safe_std = np.where(std > 1e-8, std, 1.0)
    return (matrix - mean) / safe_std


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
    distance = float(np.linalg.norm(a - b))
    return 1.0 / (1.0 + distance)


def compute_similarity_matrix(
    profiles: Sequence[SubjectProfile], config: Optional[SimilarityConfig] = None
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Pairwise similarity matrix over `profiles`' standardized summary vectors.

    Returns:
        `(matrix, subject_ids)` where `matrix[i, j]` is the similarity
        between `subject_ids[i]` and `subject_ids[j]` (symmetric,
        `matrix[i, i]` is each subject's self-similarity).
    """
    config = config or SimilarityConfig()
    if len(profiles) < 1:
        raise ValueError("Need at least 1 profile to compute a similarity matrix.")

    subject_ids = tuple(profile.subject_id for profile in profiles)
    raw = np.vstack([profile.summary_vector for profile in profiles])
    standardized = standardize_rows(raw) if len(profiles) > 1 else raw

    similarity_fn = _cosine_similarity if config.metric == "cosine" else _euclidean_similarity
    n = len(profiles)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = similarity_fn(standardized[i], standardized[j])
    return matrix, subject_ids


def find_similar_subjects(
    target_subject_id: str, profiles: Sequence[SubjectProfile], config: Optional[SimilarityConfig] = None
) -> Tuple[SimilarityResult, ...]:
    """Find the `config.top_k` subjects most similar to `target_subject_id`.

    Args:
        target_subject_id: The query subject (must be present in `profiles`).
        profiles: The population to search over (including the target).
        config: Similarity configuration; defaults applied if omitted.

    Returns:
        Up to `config.top_k` `SimilarityResult`s, excluding the target
        subject itself, sorted by `score` descending.

    Raises:
        ValueError: If `target_subject_id` is not among `profiles`.
    """
    config = config or SimilarityConfig()
    matrix, subject_ids = compute_similarity_matrix(profiles, config)
    if target_subject_id not in subject_ids:
        raise ValueError(f"'{target_subject_id}' not found among profile subject_ids: {subject_ids}")

    target_index = subject_ids.index(target_subject_id)
    results = [
        SimilarityResult(subject_id=subject_id, score=float(matrix[target_index, j]))
        for j, subject_id in enumerate(subject_ids)
        if subject_id != target_subject_id
    ]
    results.sort(key=lambda result: result.score, reverse=True)
    return tuple(results[: config.top_k])
