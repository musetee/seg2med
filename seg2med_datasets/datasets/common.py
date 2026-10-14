"""Common types and helpers for dataset adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from seg2med_datasets.schema import SCHEMA


DatasetConfig = dict[str, Any]


def catalog_config(
    name: str,
    images: Iterable[str | Path],
    segmentations: Iterable[str | Path],
    tissue_segmentations: Iterable[str | Path],
    patient_ids: Iterable[Any],
    modality: int,
) -> DatasetConfig:
    return {
        SCHEMA.CFG_NAME: name,
        SCHEMA.CFG_IMAGES: [str(path) for path in images],
        SCHEMA.CFG_SEGS: [str(path) for path in segmentations],
        SCHEMA.CFG_SEG_TISSUES: [str(path) for path in tissue_segmentations],
        SCHEMA.CFG_PIDS: [str(value) for value in patient_ids],
        SCHEMA.CFG_MODALITY: int(modality),
    }


def nifti_stem(path: Path) -> str:
    return path.name[:-7] if path.name.lower().endswith(".nii.gz") else path.stem


def with_nifti_suffix(path: Path, suffix: str) -> Path:
    extension = ".nii.gz" if path.name.lower().endswith(".nii.gz") else path.suffix
    return path.with_name(f"{nifti_stem(path)}{suffix}{extension}")
