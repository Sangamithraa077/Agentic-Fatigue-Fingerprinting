"""Object-oriented data model for the WSD4FEDSRM dataset.

See docs/data_model_architecture.md for the full design rationale and UML
diagram. This package holds only data (dataclasses); population from the
raw CSV files is the responsibility of ml.preprocessing.dataset_loader.
"""

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
from ml.data_models.sensors import EMGData, IMUData, PPGData, SensorData
from ml.data_models.subject import MVICForceProfile, Subject

__all__ = [
    "AnthropometricData",
    "Condition",
    "ConditionType",
    "Dataset",
    "DemographicData",
    "Direction",
    "EMGData",
    "EMGMuscle",
    "IMUData",
    "IMULocation",
    "IntensityBand",
    "Labels",
    "MVICForceProfile",
    "PPGData",
    "SensorData",
    "Sex",
    "Subject",
]
