"""Configurable folder scanner for NIfTI datasets."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from seg2med_workflow.config import ConfigError, resolve_path

from .common import DatasetConfig, catalog_config, nifti_stem, with_nifti_suffix

REQUIRED_CONFIG_FIELDS = (("name",),)
DEFAULT_MODALITY = 0


def _patient_id(path: Path, pattern: str | None) -> str:
    stem = nifti_stem(path)
    if not pattern:
        return stem
    match = re.search(pattern, path.as_posix())
    if not match:
        raise ConfigError(f"patient_id_regex did not match: {path}")
    if "id" in match.groupdict():
        return match.group("id")
    if match.groups():
        return match.group(1)
    return match.group(0)


def build(entry: dict[str, Any], data_root: Path) -> list[DatasetConfig]:
    name = str(entry.get("name", "")).strip()
    if not name:
        raise ConfigError("A generic dataset requires 'name'.")
    root = resolve_path(entry.get("root", "."), data_root, required=True)
    assert root is not None
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {root}")

    image_glob = str(entry.get("image_glob", "**/*.nii.gz"))
    exclude_suffixes = tuple(
        str(value) for value in entry.get("exclude_suffixes", ["_seg.nii.gz", "_seg_tissue.nii.gz"])
    )
    images = sorted(
        (
            path
            for path in root.glob(image_glob)
            if path.is_file() and not path.name.endswith(exclude_suffixes)
        ),
        key=lambda path: path.as_posix().lower(),
    )
    if not images:
        raise FileNotFoundError(f"No images matched '{image_glob}' below {root}")

    seg_suffix = str(entry.get("seg_suffix", "_seg"))
    tissue_suffix = str(entry.get("tissue_seg_suffix", "_seg_tissue"))
    segmentations = [
        str(candidate) if candidate.is_file() else ""
        for path in images
        for candidate in [with_nifti_suffix(path, seg_suffix)]
    ]
    tissue_segmentations = [
        str(candidate) if candidate.is_file() else ""
        for path in images
        for candidate in [with_nifti_suffix(path, tissue_suffix)]
    ]
    patient_ids = [_patient_id(path, entry.get("patient_id_regex")) for path in images]
    return [
        catalog_config(
            name,
            images,
            segmentations,
            tissue_segmentations,
            patient_ids,
            int(entry.get("modality", DEFAULT_MODALITY)),
        )
    ]
