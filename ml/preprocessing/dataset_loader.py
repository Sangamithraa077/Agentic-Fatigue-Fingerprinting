"""Data loading layer for the WSD4FEDSRM dataset.

This module is responsible ONLY for discovering and reading the raw files
that make up the WSD4FEDSRM dataset (Wearable Sensor Data for Fatigue
Estimation during Shoulder Rotation Movements). It performs no cleaning,
resampling, filtering, feature extraction, or modeling -- it simply locates
per-participant CSV files, classifies them by modality (EMG / IMU / PPG /
fatigue labels), and returns them as pandas DataFrames inside a structured
dictionary.

The dataset's on-disk layout is not hardcoded. Instead, participant folders
and modality files are detected via configurable naming patterns, since the
official archive organizes files hierarchically (exercise/task -> participant
-> sensor type) and exact folder names can vary slightly between dataset
versions. This keeps the loader reusable once the archive is extracted under
``datasets/raw/WSD4FEDSRM/``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, TypedDict, Union

import pandas as pd

logger = logging.getLogger(__name__)


class DatasetDirectoryNotFoundError(FileNotFoundError):
    """Raised when the configured dataset root directory does not exist."""


class ParticipantNotFoundError(ValueError):
    """Raised when a requested participant ID is not present in the dataset."""


class ParticipantData(TypedDict):
    """Structured return type for a single participant's raw data."""

    participant_id: str
    emg: Optional[pd.DataFrame]
    imu: Optional[pd.DataFrame]
    ppg: Optional[pd.DataFrame]
    labels: Optional[pd.DataFrame]
    metadata: Dict[str, Any]


# Keyword tokens used to classify a CSV file by modality, based on its file
# name and immediate parent folder name. Checked in this priority order.
_MODALITY_KEYWORDS: Dict[str, Set[str]] = {
    "emg": {"emg", "electromyography", "electromyograph", "semg"},
    "imu": {
        "imu",
        "acc",
        "accel",
        "accelerometer",
        "gyro",
        "gyroscope",
        "mag",
        "magnetometer",
        "kinematics",
    },
    "ppg": {"ppg", "photoplethysmogram", "photoplethysmography"},
    "labels": {
        "rpe",
        "borg",
        "kss",
        "karolinska",
        "fatigue",
        "label",
        "labels",
        "exertion",
    },
}
_MODALITY_PRIORITY = ("emg", "imu", "ppg", "labels")

# Matches participant folder names such as "P1", "P01", "Participant_12",
# "subject-05", "sub03", etc.
_PARTICIPANT_DIR_PATTERN = re.compile(
    r"^(?:participant|subject|sub|p)[_\-]?0*(\d+)$", re.IGNORECASE
)

_READ_CSV_ERRORS = (
    pd.errors.ParserError,
    pd.errors.EmptyDataError,
    UnicodeDecodeError,
    OSError,
)


def _tokenize(text: str) -> Set[str]:
    """Split a string into lowercase alphanumeric tokens."""
    return {token for token in re.split(r"[^a-zA-Z0-9]+", text.lower()) if token}


def _classify_file(path: Path) -> str:
    """Classify a CSV file as 'emg', 'imu', 'ppg', 'labels', or 'other'.

    Classification is based on keyword tokens found in the file name and its
    immediate parent directory name, since sensor type is typically encoded
    in one of the two within the dataset's folder hierarchy.
    """
    tokens = _tokenize(path.stem) | _tokenize(path.parent.name)
    for modality in _MODALITY_PRIORITY:
        if tokens & _MODALITY_KEYWORDS[modality]:
            return modality
    return "other"


def _normalize_participant_id(match: "re.Match[str]") -> str:
    return f"P{int(match.group(1)):02d}"


class WSD4FEDSRMLoader:
    """Reusable loader for the raw WSD4FEDSRM dataset.

    Example:
        >>> loader = WSD4FEDSRMLoader()
        >>> participants = loader.list_participants()
        >>> data = loader.load_participant(participants[0])
        >>> data["emg"].head()

    Attributes:
        root_dir: Path to the extracted WSD4FEDSRM dataset directory.
    """

    def __init__(self, root_dir: Optional[Union[str, Path]] = None) -> None:
        """Initialize the loader.

        Args:
            root_dir: Path to the extracted dataset directory. Defaults to
                ``<project_root>/datasets/raw/WSD4FEDSRM``.

        Raises:
            DatasetDirectoryNotFoundError: If the resolved directory does
                not exist or is not a directory.
        """
        self.root_dir = Path(root_dir) if root_dir is not None else self._default_root_dir()

        if not self.root_dir.exists() or not self.root_dir.is_dir():
            raise DatasetDirectoryNotFoundError(
                f"WSD4FEDSRM dataset directory not found at '{self.root_dir}'. "
                "Ensure the dataset archive has been extracted to this location."
            )

        self._global_metadata_cache: Optional[Dict[str, pd.DataFrame]] = None

    @staticmethod
    def _default_root_dir() -> Path:
        """Resolve the default dataset location relative to the project root."""
        # ml/preprocessing/dataset_loader.py -> parents[2] is the project root
        return Path(__file__).resolve().parents[2] / "datasets" / "raw" / "WSD4FEDSRM"

    def _iter_csv_files(self) -> List[Path]:
        """Recursively list every CSV file under the dataset root."""
        return sorted(self.root_dir.rglob("*.csv"))

    def _extract_participant_id(self, path: Path) -> Optional[str]:
        """Return the normalized participant ID for a path, if any ancestor
        directory name matches the participant naming pattern."""
        for part in path.parts:
            match = _PARTICIPANT_DIR_PATTERN.match(part)
            if match:
                return _normalize_participant_id(match)
        return None

    def list_participants(self) -> List[str]:
        """Discover all participant IDs present in the dataset.

        Returns:
            A sorted list of normalized participant IDs (e.g. ``["P01", "P02", ...]``).
        """
        participant_ids: Set[str] = set()
        for path in self.root_dir.rglob("*"):
            if path.is_dir():
                match = _PARTICIPANT_DIR_PATTERN.match(path.name)
                if match:
                    participant_ids.add(_normalize_participant_id(match))
        return sorted(participant_ids)

    def _gather_files(self, participant_id: str) -> Dict[str, List[Path]]:
        """Group all CSV files belonging to a participant by modality."""
        buckets: Dict[str, List[Path]] = {
            "emg": [],
            "imu": [],
            "ppg": [],
            "labels": [],
            "other": [],
        }
        for csv_path in self._iter_csv_files():
            if self._extract_participant_id(csv_path) != participant_id:
                continue
            buckets[_classify_file(csv_path)].append(csv_path)
        return buckets

    def _discover_global_files(self) -> List[Path]:
        """List CSV files not associated with any specific participant
        (e.g. dataset-wide demographics or anthropometrics tables)."""
        return [
            path
            for path in self._iter_csv_files()
            if self._extract_participant_id(path) is None
        ]

    @staticmethod
    def _read_csv(path: Path) -> Optional[pd.DataFrame]:
        """Read a single CSV file, returning None (and logging a warning) on failure."""
        try:
            return pd.read_csv(path)
        except _READ_CSV_ERRORS as exc:
            logger.warning("Skipping unreadable CSV file '%s': %s", path, exc)
            return None

    def _load_csv_group(self, paths: Sequence[Path]) -> Optional[pd.DataFrame]:
        """Read and combine a group of same-modality CSV files.

        Each row is tagged with its originating file name for traceability.
        No numeric/signal transformation is applied; files are concatenated
        as-is when more than one file exists for the same modality.
        """
        frames: List[pd.DataFrame] = []
        for path in sorted(paths):
            df = self._read_csv(path)
            if df is None:
                continue
            df = df.copy()
            df["source_file"] = path.name
            frames.append(df)

        if not frames:
            return None
        if len(frames) == 1:
            return frames[0]
        return pd.concat(frames, ignore_index=True, sort=False)

    def load_global_metadata(self) -> Dict[str, pd.DataFrame]:
        """Load dataset-wide metadata files not tied to a single participant
        (e.g. demographics, anthropometrics). Cached after the first call.

        Returns:
            A dict mapping file stem (e.g. "demographics") to its DataFrame.
        """
        if self._global_metadata_cache is None:
            cache: Dict[str, pd.DataFrame] = {}
            for path in self._discover_global_files():
                df = self._read_csv(path)
                if df is not None:
                    cache[path.stem] = df
            self._global_metadata_cache = cache
        return self._global_metadata_cache

    def load_participant(self, participant_id: str) -> ParticipantData:
        """Load all available raw data for a single participant.

        Args:
            participant_id: Normalized participant ID as returned by
                :meth:`list_participants` (e.g. ``"P01"``).

        Returns:
            A dictionary with keys ``participant_id``, ``emg``, ``imu``,
            ``ppg``, ``labels`` (each a DataFrame or ``None`` if not found),
            and ``metadata`` (a dict of supporting information).

        Raises:
            ParticipantNotFoundError: If ``participant_id`` is not present
                in the dataset.
        """
        available = self.list_participants()
        if participant_id not in available:
            raise ParticipantNotFoundError(
                f"Participant '{participant_id}' not found under '{self.root_dir}'. "
                f"Available participants: {available}"
            )

        buckets = self._gather_files(participant_id)

        emg_df = self._load_csv_group(buckets["emg"])
        imu_df = self._load_csv_group(buckets["imu"])
        ppg_df = self._load_csv_group(buckets["ppg"])
        labels_df = self._load_csv_group(buckets["labels"])

        if all(df is None for df in (emg_df, imu_df, ppg_df, labels_df)):
            logger.warning(
                "No EMG, IMU, PPG, or label files were found for participant '%s'",
                participant_id,
            )

        metadata: Dict[str, Any] = {
            "source_directory": str(self.root_dir),
            "files_found": {
                modality: [path.name for path in paths]
                for modality, paths in buckets.items()
            },
            "global_metadata": self.load_global_metadata(),
        }

        return {
            "participant_id": participant_id,
            "emg": emg_df,
            "imu": imu_df,
            "ppg": ppg_df,
            "labels": labels_df,
            "metadata": metadata,
        }

    def load_all_participants(self) -> Dict[str, ParticipantData]:
        """Load raw data for every participant discovered in the dataset.

        Individual participants that fail to load are logged and skipped
        rather than aborting the entire batch.

        Returns:
            A dict mapping participant ID to its :class:`ParticipantData`.
        """
        results: Dict[str, ParticipantData] = {}
        for participant_id in self.list_participants():
            try:
                results[participant_id] = self.load_participant(participant_id)
            except ParticipantNotFoundError as exc:  # pragma: no cover - defensive
                logger.warning("Skipping participant '%s': %s", participant_id, exc)
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        loader = WSD4FEDSRMLoader()
    except DatasetDirectoryNotFoundError as error:
        print(f"Dataset not available: {error}")
    else:
        participants = loader.list_participants()
        print(f"Discovered {len(participants)} participant(s): {participants}")

        if not participants:
            print(
                "No participant folders were detected under "
                f"'{loader.root_dir}'. Extract the WSD4FEDSRM archive there "
                "and re-run this script."
            )
        else:
            first_participant = participants[0]
            try:
                data = loader.load_participant(first_participant)
            except ParticipantNotFoundError as error:
                print(f"Failed to load participant: {error}")
            else:
                print(f"\nLoaded participant: {data['participant_id']}")
                for modality in ("emg", "imu", "ppg", "labels"):
                    frame = data[modality]
                    if frame is None:
                        print(f"  {modality}: not found")
                    else:
                        print(f"  {modality}: shape={frame.shape}, columns={list(frame.columns)}")
                print(f"  metadata keys: {list(data['metadata'].keys())}")
                print(f"  files_found: {data['metadata']['files_found']}")
