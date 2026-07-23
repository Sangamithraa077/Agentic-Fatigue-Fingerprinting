"""Top-level container representing the entire WSD4FEDSRM corpus."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from ml.data_models.subject import Subject


@dataclass(frozen=True)
class Dataset:
    """The root of the object graph: the whole WSD4FEDSRM corpus.

    Attributes:
        root_dir: Path to the extracted dataset directory.
        subjects: All loaded subjects, keyed by subject_id (up to 34).
        dataset_name: Fixed dataset identifier.
        license: Fixed license identifier.
        doi: Fixed dataset DOI, for citation/provenance.
    """

    root_dir: Path
    subjects: Dict[str, Subject] = field(default_factory=dict)
    dataset_name: str = "WSD4FEDSRM"
    license: str = "CC BY 4.0"
    doi: str = "10.5281/zenodo.8415066"
