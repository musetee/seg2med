"""Anika registered CT/MR dataset adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from seg2med_workflow.config import ConfigError, resolve_path

from .common import DatasetConfig, catalog_config, nifti_stem

REQUIRED_CONFIG_FIELDS = (("ct_root", "mr_root"),)
MODALITIES = {
    "ct": 0, "mr_vibe_in": 3, "mr_vibe_opp": 4,
    "mr_vibe_dixon_tra_in": 5, "mr_vibe_dixon_tra_opp": 6,
}


def _patient_id(stem: str) -> str:
    parts = stem.split("-")
    if len(parts) < 4:
        raise ConfigError(f"Anika filename does not contain patient/scan IDs: {stem}")
    return f"{parts[2].split('_')[0].strip()}_{parts[-1].split('_')[0].strip()}"


def _scan(root: Path) -> dict[str, dict[str, str]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Anika folder does not exist: {root}")
    scans: dict[str, dict[str, str]] = {}
    for path in sorted(root.glob("*.nii.gz")):
        stem = nifti_stem(path)
        if stem.endswith(("merged_seg", "seg_seg_tissue")):
            continue
        field = "img"
        if stem.endswith("_seg_tissue"):
            stem, field = stem[:-11], "seg_tissue"
        elif stem.endswith("_seg"):
            stem, field = stem[:-4], "seg"
        scans.setdefault(stem, {})[field] = str(path)
    return scans


def build(entry: dict[str, Any], data_root: Path) -> list[DatasetConfig]:
    ct_root = resolve_path(entry.get("ct_root"), data_root, required=True)
    mr_root = resolve_path(entry.get("mr_root"), data_root, required=True)
    assert ct_root is not None and mr_root is not None
    ct_scans, mr_scans = _scan(ct_root), _scan(mr_root)
    prefix = str(entry.get("name", "anika"))
    specs = [
        ("ct", ct_scans, MODALITIES["ct"]),
        ("mr_vibe_in", {key: row for key, row in mr_scans.items() if "vibe_in" in key}, MODALITIES["mr_vibe_in"]),
        ("mr_vibe_opp", {key: row for key, row in mr_scans.items() if "vibe_opp" in key}, MODALITIES["mr_vibe_opp"]),
        ("mr_vibe_dixon_tra_in", {key: row for key, row in mr_scans.items() if "vibe_dixon_tra_in" in key}, MODALITIES["mr_vibe_dixon_tra_in"]),
        ("mr_vibe_dixon_tra_opp", {key: row for key, row in mr_scans.items() if "vibe_dixon_tra_opp" in key}, MODALITIES["mr_vibe_dixon_tra_opp"]),
    ]
    requested = {str(value).lower() for value in entry.get("modalities", [spec[0] for spec in specs])}
    unknown = requested - {spec[0] for spec in specs}
    if unknown:
        raise ConfigError(f"Unsupported Anika modalities: {sorted(unknown)}")
    configs = []
    for key, scans, modality in specs:
        if key not in requested:
            continue
        rows = sorted(scans.items())
        configs.append(catalog_config(
            f"{prefix}_{key}",
            [files.get("img", "") for _, files in rows],
            [files.get("seg", "") for _, files in rows],
            [files.get("seg_tissue", "") for _, files in rows],
            [_patient_id(stem) for stem, _ in rows],
            modality,
        ))
    return configs
