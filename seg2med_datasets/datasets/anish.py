"""Anish CT datasets from a metadata CSV or copied ct-volume files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from seg2med_workflow.config import ConfigError, resolve_path

from . import csv_dataset
from .common import DatasetConfig, catalog_config, nifti_stem

REQUIRED_CONFIG_FIELDS = (("csv",), ("root",))
DEFAULT_MODALITY = 0
ROOT_MODALITIES = {"ct": DEFAULT_MODALITY}
CASE_PATTERN = re.compile(r"ct-volume-(?P<id>[^_]+)(?P<suffix>_seg_tissue|_seg)?$", re.IGNORECASE)


def build(entry: dict[str, Any], data_root: Path) -> list[DatasetConfig]:
    # Keep the existing CSV behavior, including column mappings and paths.
    if entry.get("csv"):
        return csv_dataset.build(entry, data_root)

    if not entry.get("root"):
        raise ConfigError("An Anish dataset requires either 'csv' or 'root'.")
    root = resolve_path(entry["root"], data_root, required=True)
    assert root is not None
    if not root.is_dir():
        raise FileNotFoundError(f"Anish folder does not exist: {root}")
    requested = entry.get("modalities", list(ROOT_MODALITIES))
    if not isinstance(requested, list) or {str(value).lower() for value in requested} != set(ROOT_MODALITIES):
        raise ConfigError("Anish root scanning supports modalities: [ct].")
    if int(entry.get("modality", DEFAULT_MODALITY)) != DEFAULT_MODALITY:
        raise ConfigError("Anish ct-volume root scanning uses modality: 0 (ct).")

    scans: dict[tuple[Path, str], dict[str, str]] = {}
    fields = {None: "img", "_seg": "seg", "_seg_tissue": "seg_tissue"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not path.name.lower().endswith((".nii", ".nii.gz")):
            continue
        match = CASE_PATTERN.fullmatch(nifti_stem(path))
        if match is None:
            continue
        suffix = match["suffix"].lower() if match["suffix"] else None
        field = fields[suffix]
        row = scans.setdefault((path.parent, match["id"]), {})
        if field in row:
            raise ConfigError(f"Multiple Anish {field} files for case {match['id']}: {row[field]} and {path}")
        row[field] = str(path)
    if not scans:
        raise FileNotFoundError(f"No ct-volume-*.nii or ct-volume-*.nii.gz case files found below {root}")

    rows = sorted(scans.items())
    return [catalog_config(
        str(entry.get("name", "anish")),
        [files.get("img", "") for _, files in rows],
        [files.get("seg", "") for _, files in rows],
        [files.get("seg_tissue", "") for _, files in rows],
        [patient_id for (_, patient_id), _ in rows],
        DEFAULT_MODALITY,
    )]
