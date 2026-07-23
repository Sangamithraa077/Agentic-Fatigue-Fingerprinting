"""Data loading layer for the WSD4FEDSRM dataset.

This module discovers and reads the raw WSD4FEDSRM files on disk and
constructs the typed object graph defined in ``ml.data_models``
(Dataset -> Subject -> Condition -> {EMGData, IMUData, PPGData, Labels}).
It performs no signal processing, normalization, feature extraction, or
modeling -- those are the responsibilities of later phases, which will
consume these objects and produce new ones rather than mutating them.

Subjects and experimental conditions are discovered dynamically from the
directory tree (never hardcoded): subject folders are matched with a
"Subject <n>" pattern, and condition folders are matched against two
naming patterns (fatigue tasks and MVIC calibration trials) documented in
the dataset's own description.txt. The one thing that *is* fixed is the
mapping from (direction, intensity band) to the dataset's internal Borg
task code (e.g. "task4_35e") -- description.txt confirms these codes are
assigned unvarying regardless of subject, so this mapping is a documented
protocol constant, not a hardcoded subject/condition name.

Run as a script (``python ml/preprocessing/dataset_loader.py`` or
``python -m ml.preprocessing.dataset_loader``) to load the dataset and
print summary statistics.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import pandas as pd

# Allow `python ml/preprocessing/dataset_loader.py` to work even though it
# uses absolute `ml.*` imports below (which require the project root to be
# on sys.path). Running as `python -m ml.preprocessing.dataset_loader` from
# the project root does not need this.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.data_models.anthropometry import AnthropometricData
from ml.data_models.condition import Condition
from ml.data_models.dataset import Dataset
from ml.data_models.demographics import DemographicData
from ml.data_models.enums import (
    ConditionType,
    Direction,
    EMGMuscle,
    IMULocation,
    IntensityBand,
    Sex,
)
from ml.data_models.labels import Labels
from ml.data_models.sensors import EMGData, IMUData, PPGData
from ml.data_models.subject import MVICForceProfile, Subject

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Confirmed sampling rates (description.txt / published paper) -- these are
# fixed instrument properties of the study protocol, not discoverable from
# the raw CSVs themselves (the files carry no time column).
# --------------------------------------------------------------------------
EMG_SAMPLING_RATE_HZ: float = 1000.0
IMU_SAMPLING_RATE_HZ: float = 100.0
PPG_SAMPLING_RATE_HZ: float = 200.0


class DatasetDirectoryNotFoundError(FileNotFoundError):
    """Raised when the configured dataset root directory does not exist."""


class NoSubjectsFoundError(RuntimeError):
    """Raised when zero subject folders can be discovered under the dataset root."""


# --------------------------------------------------------------------------
# Fixed protocol-level mapping: (direction, intensity band) -> Borg task code.
# Source: WSD4FEDSRM/description.txt -- "task1_35i .. task6_55e" labels are
# assigned in a fixed order regardless of subject; the per-subject 'group'
# column only records chronological execution order for counterbalancing.
# --------------------------------------------------------------------------
_TASK_ORDER_CODE_BY_SPEC: Dict[Tuple[Direction, IntensityBand], str] = {
    (Direction.INTERNAL_ROTATION, IntensityBand.BAND_30_40): "task1_35i",
    (Direction.INTERNAL_ROTATION, IntensityBand.BAND_40_50): "task2_45i",
    (Direction.INTERNAL_ROTATION, IntensityBand.BAND_50_60): "task3_55i",
    (Direction.EXTERNAL_ROTATION, IntensityBand.BAND_30_40): "task4_35e",
    (Direction.EXTERNAL_ROTATION, IntensityBand.BAND_40_50): "task5_45e",
    (Direction.EXTERNAL_ROTATION, IntensityBand.BAND_50_60): "task6_55e",
}

_SUBJECT_FOLDER_PATTERN = re.compile(r"^Subject\s+(\d+)$", re.IGNORECASE)
_SUBJECT_VALUE_PATTERN = re.compile(r"subject[_\s]*?(\d+)", re.IGNORECASE)
_FATIGUE_CONDITION_PATTERN = re.compile(
    r"^(\d{2})-(\d{2})_\s*(internal|external)\s*rotation$", re.IGNORECASE
)
_MVIC_CONDITION_PATTERN = re.compile(
    r"^MVIC force (internal|external) rotation (first|second)$", re.IGNORECASE
)
_RPE_TIMESERIES_COLUMN_PATTERN = re.compile(r"^(\d+)_sec$", re.IGNORECASE)

_READ_CSV_ERRORS = (
    pd.errors.ParserError,
    pd.errors.EmptyDataError,
    UnicodeDecodeError,
    OSError,
)


@dataclass(frozen=True)
class _ConditionSpec:
    """Internal helper: the parsed meaning of a condition folder name."""

    condition_type: ConditionType
    direction: Direction
    intensity_band: Optional[IntensityBand]
    trial_number: Optional[int]
    task_order_code: Optional[str]


def _parse_condition_folder(folder_name: str) -> Optional[_ConditionSpec]:
    """Parse a condition folder name into a `_ConditionSpec`, or None if
    the name doesn't match either known naming pattern."""
    name = folder_name.strip()

    match = _FATIGUE_CONDITION_PATTERN.match(name)
    if match:
        band_lo, band_hi, direction_word = match.group(1), match.group(2), match.group(3).lower()
        direction = Direction.INTERNAL_ROTATION if direction_word == "internal" else Direction.EXTERNAL_ROTATION
        try:
            intensity_band = IntensityBand(f"{band_lo}-{band_hi}")
        except ValueError:
            return None
        task_order_code = _TASK_ORDER_CODE_BY_SPEC.get((direction, intensity_band))
        return _ConditionSpec(ConditionType.FATIGUE_TASK, direction, intensity_band, None, task_order_code)

    match = _MVIC_CONDITION_PATTERN.match(name)
    if match:
        direction_word, trial_word = match.group(1).lower(), match.group(2).lower()
        direction = Direction.INTERNAL_ROTATION if direction_word == "internal" else Direction.EXTERNAL_ROTATION
        trial_number = 1 if trial_word == "first" else 2
        return _ConditionSpec(ConditionType.MVIC_CALIBRATION, direction, None, trial_number, None)

    return None


def _extract_subject_number(text: str) -> Optional[int]:
    """Extract the numeric subject id from either a folder name
    ("Subject 1") or a CSV cell value ("subject_1")."""
    match = _SUBJECT_VALUE_PATTERN.search(text)
    return int(match.group(1)) if match else None


def _canonical_subject_id(number: int) -> str:
    return f"P{number:02d}"


def _normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _find_column(columns: Iterable[str], *tokens: str) -> Optional[str]:
    """Find the first column whose normalized name contains every given
    token, tolerant of the source files' inconsistent spacing/punctuation
    (e.g. "IR_MVIC_2_ (N)"). Returns None if no column matches."""
    normalized_tokens = [_normalize_token(token) for token in tokens]
    for column in columns:
        normalized_column = _normalize_token(column)
        if all(token in normalized_column for token in normalized_tokens):
            return column
    return None


def _require_column(columns: Iterable[str], *tokens: str) -> str:
    column = _find_column(columns, *tokens)
    if column is None:
        raise KeyError(f"No column matching tokens {tokens!r} found among {list(columns)!r}")
    return column


class WSD4FEDSRMLoader:
    """Reusable loader that builds the typed WSD4FEDSRM object graph.

    Example:
        >>> loader = WSD4FEDSRMLoader()
        >>> dataset = loader.load_dataset()
        >>> dataset.subjects["P01"].conditions["P01_external_rotation_30-40"].emg
    """

    # Fixed top-level folder names -- these are the dataset publisher's own
    # structural layout (confirmed against the official Zenodo archive), not
    # subject/condition names, so they are legitimate named constants.
    _EMG_IMU_PPG_DIRNAME = "EMG, IMU, and PPG data"
    _BORG_DIRNAME = "Borg data"
    _KSS_DIRNAME = "KSS data"
    _MVIC_DIRNAME = "MVIC force data"
    _DEMOGRAPHIC_DIRNAME = "Demographic and antropometric data"

    def __init__(self, root_dir: Optional[Union[str, Path]] = None) -> None:
        """Initialize the loader.

        Args:
            root_dir: Path to the extracted WSD4FEDSRM directory. Defaults
                to ``<project_root>/datasets/raw/WSD4FEDSRM``.

        Raises:
            DatasetDirectoryNotFoundError: If the resolved directory does
                not exist.
        """
        self.root_dir = Path(root_dir) if root_dir is not None else self._default_root_dir()
        if not self.root_dir.is_dir():
            raise DatasetDirectoryNotFoundError(
                f"WSD4FEDSRM dataset directory not found at '{self.root_dir}'. "
                "Ensure the dataset archive has been extracted to this location."
            )

        self._borg_cache: Optional[pd.DataFrame] = None
        self._kss_cache: Optional[pd.DataFrame] = None
        self._mvic_cache: Optional[pd.DataFrame] = None
        self._demographic_cache: Optional[pd.DataFrame] = None
        self._anthro_cache: Dict[str, pd.DataFrame] = {}

    @staticmethod
    def _default_root_dir() -> Path:
        return Path(__file__).resolve().parents[2] / "datasets" / "raw" / "WSD4FEDSRM"

    # ----------------------------------------------------------------
    # Discovery (no hardcoded subject/condition names)
    # ----------------------------------------------------------------

    def discover_subjects(self) -> Dict[str, str]:
        """Discover all subject folders present in the dataset.

        Returns:
            Mapping of canonical subject id (e.g. "P01") to the exact raw
            folder name on disk (e.g. "Subject 1"), sorted by subject id.
        """
        subjects: Dict[str, str] = {}
        base_dir = self.root_dir / self._EMG_IMU_PPG_DIRNAME
        if not base_dir.is_dir():
            logger.error("Sensor data directory not found: %s", base_dir)
            return subjects

        for condition_dir in base_dir.iterdir():
            if not condition_dir.is_dir():
                continue
            for subject_dir in condition_dir.iterdir():
                if not subject_dir.is_dir():
                    continue
                match = _SUBJECT_FOLDER_PATTERN.match(subject_dir.name.strip())
                if match:
                    canonical_id = _canonical_subject_id(int(match.group(1)))
                    subjects.setdefault(canonical_id, subject_dir.name)

        return dict(sorted(subjects.items()))

    def discover_conditions(self) -> Dict[str, _ConditionSpec]:
        """Discover all experimental condition folders present in the dataset.

        Returns:
            Mapping of raw condition folder name to its parsed
            `_ConditionSpec`. Folder names that match neither known naming
            pattern are logged and skipped.
        """
        conditions: Dict[str, _ConditionSpec] = {}
        base_dir = self.root_dir / self._EMG_IMU_PPG_DIRNAME
        if not base_dir.is_dir():
            logger.error("Sensor data directory not found: %s", base_dir)
            return conditions

        for condition_dir in sorted(base_dir.iterdir()):
            if not condition_dir.is_dir():
                continue
            spec = _parse_condition_folder(condition_dir.name)
            if spec is None:
                logger.warning("Unrecognized condition folder name '%s' -- skipping", condition_dir.name)
                continue
            conditions[condition_dir.name] = spec

        return conditions

    # ----------------------------------------------------------------
    # Top-level orchestration
    # ----------------------------------------------------------------

    def load_dataset(self) -> Dataset:
        """Load the entire WSD4FEDSRM dataset into a typed `Dataset` object.

        Returns:
            A `Dataset` populated with every subject that could be loaded.

        Raises:
            NoSubjectsFoundError: If no subject folders could be discovered
                at all (a fatal, dataset-wide problem).
        """
        logger.info("Discovering subjects and conditions under '%s'", self.root_dir)
        subject_map = self.discover_subjects()
        condition_map = self.discover_conditions()
        logger.info(
            "Discovered %d subject(s) and %d condition(s)", len(subject_map), len(condition_map)
        )

        if not subject_map:
            raise NoSubjectsFoundError(
                f"No subject folders were discovered under '{self.root_dir}'. "
                "Verify the dataset was extracted correctly."
            )
        if len(condition_map) != 10:
            logger.warning(
                "Expected 10 experimental conditions (6 fatigue + 4 MVIC), found %d",
                len(condition_map),
            )

        subjects: Dict[str, Subject] = {}
        for subject_id, raw_folder_name in subject_map.items():
            try:
                subjects[subject_id] = self.load_subject(subject_id, raw_folder_name, condition_map)
            except Exception:
                logger.exception("Failed to load subject '%s' -- skipping", subject_id)

        if len(subjects) != len(subject_map):
            logger.warning(
                "Only %d/%d discovered subjects were loaded successfully",
                len(subjects),
                len(subject_map),
            )

        return Dataset(root_dir=self.root_dir, subjects=subjects)

    def load_subject(
        self,
        subject_id: str,
        raw_folder_name: str,
        condition_map: Dict[str, _ConditionSpec],
    ) -> Subject:
        """Load one subject: their profile, session outcomes, and every
        condition trial they performed.

        Args:
            subject_id: Canonical subject id (e.g. "P01").
            raw_folder_name: The exact on-disk folder name (e.g. "Subject 1").
            condition_map: All discovered conditions, from `discover_conditions`.

        Returns:
            A fully populated `Subject`. Sub-components that fail to load
            are logged and set to None/empty rather than aborting the load.
        """
        demographic = self.load_demographics(subject_id)
        anthropometric = self.load_anthropometrics(subject_id)
        kss_before, kss_after = self._load_kss(subject_id)
        mvic_force = self._load_mvic_force(subject_id)

        conditions: Dict[str, Condition] = {}
        for condition_folder_name, spec in condition_map.items():
            try:
                condition = self.load_condition(subject_id, raw_folder_name, condition_folder_name, spec)
                conditions[condition.condition_id] = condition
            except Exception:
                logger.exception(
                    "Failed to load condition '%s' for subject '%s' -- skipping",
                    condition_folder_name,
                    subject_id,
                )

        if len(conditions) != len(condition_map):
            logger.warning(
                "Subject '%s': only %d/%d conditions loaded successfully",
                subject_id,
                len(conditions),
                len(condition_map),
            )
        if demographic is None:
            logger.warning("Subject '%s': demographic data unavailable", subject_id)
        if anthropometric is None:
            logger.warning("Subject '%s': anthropometric data unavailable", subject_id)
        if mvic_force is None:
            logger.warning("Subject '%s': MVIC force data unavailable", subject_id)

        return Subject(
            subject_id=subject_id,
            demographic=demographic,
            anthropometric=anthropometric,
            mvic_force=mvic_force,
            kss_before=kss_before,
            kss_after=kss_after,
            conditions=conditions,
        )

    def load_condition(
        self,
        subject_id: str,
        raw_folder_name: str,
        condition_folder_name: str,
        spec: _ConditionSpec,
    ) -> Condition:
        """Load one subject's single trial (fatigue task or MVIC trial).

        Args:
            subject_id: Canonical subject id (e.g. "P01").
            raw_folder_name: The subject's exact on-disk folder name.
            condition_folder_name: The exact on-disk condition folder name.
            spec: The parsed condition specification for this folder.

        Returns:
            A fully populated `Condition`, with missing modalities set to
            None/empty (e.g. `ppg=None` for MVIC trials) rather than raising.

        Raises:
            FileNotFoundError: If the expected subject-within-condition
                directory does not exist at all.
        """
        condition_dir = self.root_dir / self._EMG_IMU_PPG_DIRNAME / condition_folder_name / raw_folder_name
        if not condition_dir.is_dir():
            raise FileNotFoundError(f"Expected condition directory not found: {condition_dir}")

        if spec.condition_type is ConditionType.FATIGUE_TASK:
            condition_id = f"{subject_id}_{spec.direction.value}_{spec.intensity_band.value}"
        else:
            condition_id = f"{subject_id}_mvic_{spec.direction.value}_trial{spec.trial_number}"

        emg = self.load_emg(condition_dir, subject_id, condition_id)
        imu = self.load_imu(condition_dir, subject_id, condition_id)
        ppg = self.load_ppg(condition_dir, subject_id, condition_id)
        labels = (
            self.load_labels(subject_id, spec.task_order_code)
            if spec.condition_type is ConditionType.FATIGUE_TASK
            else None
        )

        if len(emg) < len(EMGMuscle):
            logger.warning(
                "Condition '%s': only %d/%d EMG channels found", condition_id, len(emg), len(EMGMuscle)
            )
        if len(imu) < len(IMULocation):
            logger.warning(
                "Condition '%s': only %d/%d IMU locations found", condition_id, len(imu), len(IMULocation)
            )

        return Condition(
            condition_id=condition_id,
            subject_id=subject_id,
            condition_type=spec.condition_type,
            direction=spec.direction,
            intensity_band=spec.intensity_band,
            trial_number=spec.trial_number,
            task_order_code=spec.task_order_code,
            emg=emg,
            imu=imu,
            ppg=ppg,
            labels=labels,
        )

    # ----------------------------------------------------------------
    # Sensor loading
    # ----------------------------------------------------------------

    def load_emg(self, condition_dir: Path, subject_id: str, condition_id: str) -> Dict[EMGMuscle, EMGData]:
        """Load all available EMG channel files for one trial.

        Missing or unreadable files are logged and skipped; this method
        never raises for a single bad file.
        """
        emg_dir = condition_dir / "EMG data"
        result: Dict[EMGMuscle, EMGData] = {}
        if not emg_dir.is_dir():
            logger.warning("EMG data folder missing: %s", emg_dir)
            return result

        for csv_path in sorted(emg_dir.glob("*.csv")):
            try:
                muscle = EMGMuscle(csv_path.stem.strip().lower())
            except ValueError:
                logger.warning("Unrecognized EMG muscle file '%s' -- skipping", csv_path.name)
                continue
            signal = self._read_csv(csv_path)
            if signal is None:
                continue
            result[muscle] = EMGData(
                subject_id=subject_id,
                condition_id=condition_id,
                sampling_rate_hz=EMG_SAMPLING_RATE_HZ,
                source_file=csv_path,
                muscle=muscle,
                signal=signal,
            )
        return result

    def load_imu(self, condition_dir: Path, subject_id: str, condition_id: str) -> Dict[IMULocation, IMUData]:
        """Load all available IMU location recordings (acc+gyro+mag) for
        one trial.

        Missing or unreadable locations are logged and skipped; this
        method never raises for a single bad location.
        """
        imu_dir = condition_dir / "IMU data"
        result: Dict[IMULocation, IMUData] = {}
        if not imu_dir.is_dir():
            logger.warning("IMU data folder missing: %s", imu_dir)
            return result

        for location_dir in sorted(imu_dir.iterdir()):
            if not location_dir.is_dir():
                continue
            normalized = location_dir.name.strip().lower().replace(" ", "_")
            try:
                location = IMULocation(normalized)
            except ValueError:
                logger.warning("Unrecognized IMU location folder '%s' -- skipping", location_dir.name)
                continue

            accelerometer = self._read_csv(location_dir / f"acc_{normalized}.csv")
            gyroscope = self._read_csv(location_dir / f"gyr_{normalized}.csv")
            magnetometer = self._read_csv(location_dir / f"mag_{normalized}.csv")
            if accelerometer is None or gyroscope is None or magnetometer is None:
                logger.warning(
                    "Incomplete IMU stream set for location '%s' in %s -- skipping", location, location_dir
                )
                continue

            result[location] = IMUData(
                subject_id=subject_id,
                condition_id=condition_id,
                sampling_rate_hz=IMU_SAMPLING_RATE_HZ,
                source_file=location_dir,
                location=location,
                accelerometer=accelerometer,
                gyroscope=gyroscope,
                magnetometer=magnetometer,
            )
        return result

    def load_ppg(self, condition_dir: Path, subject_id: str, condition_id: str) -> Optional[PPGData]:
        """Load the PPG recording for one trial, if present.

        Returns None both when PPG is expectedly absent (MVIC trials) and
        when a PPG file exists but fails to read (logged as a warning) --
        callers should not treat None as necessarily an error.
        """
        ppg_path = condition_dir / "PPG data" / "ppg.csv"
        if not ppg_path.is_file():
            return None

        signal = self._read_csv(ppg_path)
        if signal is None:
            return None

        return PPGData(
            subject_id=subject_id,
            condition_id=condition_id,
            sampling_rate_hz=PPG_SAMPLING_RATE_HZ,
            source_file=ppg_path,
            signal=signal,
        )

    @staticmethod
    def _read_csv(path: Path) -> Optional[pd.DataFrame]:
        """Read a single CSV file, returning None (and logging) on failure."""
        try:
            return pd.read_csv(path)
        except _READ_CSV_ERRORS as exc:
            logger.warning("Failed to read CSV file '%s': %s", path, exc)
            return None
        except FileNotFoundError:
            logger.warning("Expected CSV file not found: '%s'", path)
            return None

    # ----------------------------------------------------------------
    # Labels
    # ----------------------------------------------------------------

    def load_labels(self, subject_id: str, task_order_code: Optional[str]) -> Optional[Labels]:
        """Load the Borg RPE fatigue trajectory for one fatigue-task trial.

        Args:
            subject_id: Canonical subject id (e.g. "P01").
            task_order_code: The Borg task code for this trial (e.g.
                "task4_35e"), or None for MVIC trials (which have no Borg
                data, and for which this method always returns None).

        Returns:
            The `Labels` for this trial, or None if not applicable / not found.
        """
        if task_order_code is None:
            return None

        df = self._borg_df()
        if df is None:
            return None

        subject_col = _find_column(df.columns, "subject")
        task_col = _find_column(df.columns, "task", "order")
        if subject_col is None or task_col is None:
            logger.warning("Borg data file is missing expected 'subject'/'task_order' columns")
            return None

        subject_numbers = df[subject_col].apply(lambda v: _extract_subject_number(str(v)) if pd.notna(v) else None)
        mask = (subject_numbers.apply(lambda n: n is not None and _canonical_subject_id(n) == subject_id)) & (
            df[task_col].astype(str).str.strip() == task_order_code
        )
        matches = df[mask]
        if matches.empty:
            logger.warning("No Borg RPE row found for subject '%s', task '%s'", subject_id, task_order_code)
            return None

        row = matches.iloc[0]
        before_col = _find_column(df.columns, "before", "task")
        end_col = _find_column(df.columns, "end", "trial")
        length_col = _find_column(df.columns, "length", "trial")

        rpe_timeseries: Dict[int, int] = {}
        for column in df.columns:
            match = _RPE_TIMESERIES_COLUMN_PATTERN.match(column.strip())
            if not match:
                continue
            value = row[column]
            if pd.notna(value):
                try:
                    rpe_timeseries[int(match.group(1))] = int(value)
                except (TypeError, ValueError):
                    continue

        def _safe_int(col: Optional[str]) -> Optional[int]:
            if col is None or pd.isna(row[col]):
                return None
            try:
                return int(row[col])
            except (TypeError, ValueError):
                return None

        def _safe_float(col: Optional[str]) -> Optional[float]:
            if col is None or pd.isna(row[col]):
                return None
            try:
                return float(row[col])
            except (TypeError, ValueError):
                return None

        return Labels(
            task_order_code=task_order_code,
            rpe_before=_safe_int(before_col),
            rpe_timeseries=rpe_timeseries,
            rpe_at_end_of_trial=_safe_int(end_col),
            trial_duration_seconds=_safe_float(length_col),
        )

    def _borg_df(self) -> Optional[pd.DataFrame]:
        if self._borg_cache is None:
            path = self.root_dir / self._BORG_DIRNAME / "borg_data.csv"
            df = self._read_csv(path)
            if df is None:
                return None
            df.columns = [column.strip() for column in df.columns]
            subject_col = _find_column(df.columns, "subject")
            if subject_col is not None:
                # The source file only fills 'subject' on the first of each
                # subject's 6 task rows; forward-fill to associate every row
                # with its subject.
                df[subject_col] = df[subject_col].ffill()
            self._borg_cache = df
        return self._borg_cache

    # ----------------------------------------------------------------
    # Subject-level metadata
    # ----------------------------------------------------------------

    def load_demographics(self, subject_id: str) -> Optional[DemographicData]:
        """Load one subject's demographic profile from demographic.csv."""
        df = self._demographic_df()
        if df is None:
            return None
        row = self._row_for_subject(df, subject_id)
        if row is None:
            logger.warning("No demographic record found for subject '%s'", subject_id)
            return None

        try:
            age_col = _require_column(df.columns, "age")
            height_col = _require_column(df.columns, "height")
            sex_col = _require_column(df.columns, "sex")
            hand_col = _require_column(df.columns, "dominant")
            group_col = _require_column(df.columns, "group")
            exercise_type_col = _find_column(df.columns, "kind", "exercise") or _find_column(df.columns, "partake")
            exercise_freq_col = _find_column(df.columns, "often") or _find_column(df.columns, "week")
            date_col = _find_column(df.columns, "date")
            time_col = _find_column(df.columns, "time")

            return DemographicData(
                age=int(row[age_col]),
                height_cm=float(row[height_col]),
                sex=Sex(str(row[sex_col]).strip().lower()),
                dominant_hand=str(row[hand_col]).strip().lower(),
                execution_order_group=int(row[group_col]),
                exercise_type=str(row[exercise_type_col]).strip() if exercise_type_col else "",
                exercise_frequency_per_week=(
                    int(row[exercise_freq_col])
                    if exercise_freq_col and pd.notna(row[exercise_freq_col])
                    else 0
                ),
                assessment_date=str(row[date_col]).strip() if date_col and pd.notna(row[date_col]) else None,
                assessment_time=str(row[time_col]).strip() if time_col and pd.notna(row[time_col]) else None,
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Failed to parse demographic record for subject '%s': %s", subject_id, exc)
            return None

    def load_anthropometrics(self, subject_id: str) -> Optional[AnthropometricData]:
        """Load one subject's anthropometric profile, merged from
        body_composition.csv, breadth.csv, circumference.csv, depth.csv,
        and length.csv."""
        try:
            body_df = self._anthro_df("body_composition.csv")
            breadth_df = self._anthro_df("breadth.csv")
            circumference_df = self._anthro_df("circumference.csv")
            depth_df = self._anthro_df("depth.csv")
            length_df = self._anthro_df("length.csv")

            if any(df is None for df in (body_df, breadth_df, circumference_df, depth_df, length_df)):
                logger.warning("One or more anthropometric source files could not be read")
                return None

            body_row = self._row_for_subject(body_df, subject_id)
            breadth_row = self._row_for_subject(breadth_df, subject_id)
            circumference_row = self._row_for_subject(circumference_df, subject_id)
            depth_row = self._row_for_subject(depth_df, subject_id)
            length_row = self._row_for_subject(length_df, subject_id)

            if any(row is None for row in (body_row, breadth_row, circumference_row, depth_row, length_row)):
                logger.warning("Incomplete anthropometric records for subject '%s'", subject_id)
                return None

            mass_col = _require_column(body_df.columns, "mass")
            fat_col = _require_column(body_df.columns, "body", "fat")
            muscle_col = _require_column(body_df.columns, "muscle")
            visceral_col = _require_column(body_df.columns, "visceral")
            bmi_col = _require_column(body_df.columns, "bmi")

            return AnthropometricData(
                mass_kg=float(body_row[mass_col]),
                body_fat_percent=float(body_row[fat_col]),
                muscle_percent=float(body_row[muscle_col]),
                visceral_fat=float(body_row[visceral_col]),
                bmi=float(body_row[bmi_col]),
                breadth_cm=self._measurement_dict(breadth_df, breadth_row),
                circumference_cm=self._measurement_dict(circumference_df, circumference_row),
                depth_cm=self._measurement_dict(depth_df, depth_row),
                length_cm=self._measurement_dict(length_df, length_row),
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Failed to parse anthropometric data for subject '%s': %s", subject_id, exc)
            return None

    @staticmethod
    def _measurement_dict(df: pd.DataFrame, row: pd.Series) -> Dict[str, float]:
        """Build a {original_column_name: value} dict for every non-subject,
        non-null column in a row (used for breadth/circumference/depth/length,
        whose column sets are only known at read time)."""
        subject_col = _find_column(df.columns, "subject")
        result: Dict[str, float] = {}
        for column in df.columns:
            if column == subject_col:
                continue
            value = row[column]
            if pd.notna(value):
                try:
                    result[column] = float(value)
                except (TypeError, ValueError):
                    continue
        return result

    def _load_kss(self, subject_id: str) -> Tuple[Optional[int], Optional[int]]:
        df = self._kss_df()
        if df is None:
            return None, None
        row = self._row_for_subject(df, subject_id)
        if row is None:
            logger.warning("No KSS record found for subject '%s'", subject_id)
            return None, None
        try:
            before_col = _require_column(df.columns, "before")
            after_col = _require_column(df.columns, "after")
            return int(row[before_col]), int(row[after_col])
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Failed to parse KSS record for subject '%s': %s", subject_id, exc)
            return None, None

    def _load_mvic_force(self, subject_id: str) -> Optional[MVICForceProfile]:
        df = self._mvic_df()
        if df is None:
            return None
        row = self._row_for_subject(df, subject_id)
        if row is None:
            logger.warning("No MVIC force record found for subject '%s'", subject_id)
            return None
        try:
            ir1_col = _require_column(df.columns, "ir", "1")
            ir2_col = _require_column(df.columns, "ir", "2")
            ir_mean_col = _require_column(df.columns, "ir", "mean")
            er1_col = _require_column(df.columns, "er", "1")
            er2_col = _require_column(df.columns, "er", "2")
            er_mean_col = _require_column(df.columns, "er", "mean")
            return MVICForceProfile(
                ir_trial_1_n=float(row[ir1_col]),
                ir_trial_2_n=float(row[ir2_col]),
                ir_mean_n=float(row[ir_mean_col]),
                er_trial_1_n=float(row[er1_col]),
                er_trial_2_n=float(row[er2_col]),
                er_mean_n=float(row[er_mean_col]),
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Failed to parse MVIC force record for subject '%s': %s", subject_id, exc)
            return None

    def _row_for_subject(self, df: pd.DataFrame, subject_id: str) -> Optional[pd.Series]:
        """Find the row in a subject-indexed DataFrame matching the given
        canonical subject id, regardless of the raw 'subject'-column
        value's exact formatting (e.g. "subject_1")."""
        subject_col = _find_column(df.columns, "subject")
        if subject_col is None:
            return None
        for _, row in df.iterrows():
            value = row[subject_col]
            if pd.isna(value):
                continue
            number = _extract_subject_number(str(value))
            if number is not None and _canonical_subject_id(number) == subject_id:
                return row
        return None

    def _demographic_df(self) -> Optional[pd.DataFrame]:
        if self._demographic_cache is None:
            path = self.root_dir / self._DEMOGRAPHIC_DIRNAME / "demographic.csv"
            df = self._read_csv(path)
            if df is None:
                return None
            df.columns = [column.strip() for column in df.columns]
            self._demographic_cache = df
        return self._demographic_cache

    def _anthro_df(self, filename: str) -> Optional[pd.DataFrame]:
        if filename not in self._anthro_cache:
            path = self.root_dir / self._DEMOGRAPHIC_DIRNAME / filename
            df = self._read_csv(path)
            if df is None:
                return None
            df.columns = [column.strip() for column in df.columns]
            self._anthro_cache[filename] = df
        return self._anthro_cache.get(filename)

    def _kss_df(self) -> Optional[pd.DataFrame]:
        if self._kss_cache is None:
            path = self.root_dir / self._KSS_DIRNAME / "KSS_data.csv"
            df = self._read_csv(path)
            if df is None:
                return None
            df.columns = [column.strip() for column in df.columns]
            self._kss_cache = df
        return self._kss_cache

    def _mvic_df(self) -> Optional[pd.DataFrame]:
        if self._mvic_cache is None:
            path = self.root_dir / self._MVIC_DIRNAME / "MVIC_force_data.csv"
            df = self._read_csv(path)
            if df is None:
                return None
            df.columns = [column.strip() for column in df.columns]
            self._mvic_cache = df
        return self._mvic_cache


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        loader = WSD4FEDSRMLoader()
        dataset = loader.load_dataset()
    except (DatasetDirectoryNotFoundError, NoSubjectsFoundError) as error:
        print(f"Dataset could not be loaded: {error}")
    else:
        condition_catalog = loader.discover_conditions()
        ppg_available = any(
            condition.ppg is not None
            for subject in dataset.subjects.values()
            for condition in subject.conditions.values()
        )

        print("\nDataset Summary")
        print(f"Subjects: {len(dataset.subjects)}")
        print(f"Conditions: {len(condition_catalog)}")
        print(f"EMG Channels: {len(EMGMuscle)}")
        print(f"IMU Sensors: {len(IMULocation)}")
        print(f"PPG Available: {'Yes' if ppg_available else 'No'}")

        first_subject_id = next(iter(dataset.subjects), None)
        if first_subject_id is not None:
            subject = dataset.subjects[first_subject_id]
            print(f"\nExample subject: {subject.subject_id} ({len(subject.conditions)} conditions loaded)")
            example_condition_id = next(iter(subject.conditions), None)
            if example_condition_id is not None:
                condition = subject.conditions[example_condition_id]
                print(
                    f"Example condition: {condition.condition_id} "
                    f"(EMG={len(condition.emg)}, IMU={len(condition.imu)}, "
                    f"PPG={'yes' if condition.ppg else 'no'}, "
                    f"Labels={'yes' if condition.labels else 'no'})"
                )
