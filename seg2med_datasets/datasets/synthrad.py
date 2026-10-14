"""SynthRAD and trial-layout dataset adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from seg2med_workflow.config import ConfigError, resolve_path

from .common import DatasetConfig, catalog_config, nifti_stem

REQUIRED_CONFIG_FIELDS = (("root",),)
MODALITIES = {"ct": 0, "mr_t1": 1, "mr_t2": 2}


def build(entry: dict[str, Any], data_root: Path) -> list[DatasetConfig]:
    root = resolve_path(entry.get("root"), data_root, required=True)
    assert root is not None
    if not root.is_dir():
        raise FileNotFoundError(f"SynthRAD folder does not exist: {root}")
    default_name = "trial" if str(entry.get("kind", "")).lower() == "trial" else "synthrad"
    prefix = str(entry.get("name", default_name))
    requested = {str(value).lower() for value in entry.get("modalities", list(MODALITIES))}
    known = set(MODALITIES)
    if requested - known:
        raise ConfigError(f"Unsupported SynthRAD modalities: {sorted(requested - known)}")
    scans: dict[str, dict[Path, dict[str, str]]] = {key: {} for key in known}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not path.name.lower().endswith((".nii.gz", ".nii", ".nrrd")):
            continue
        stem = nifti_stem(path).lower()
        field = "img"
        if stem.endswith("_seg_tissue"):
            stem, field = stem[:-11], "seg_tissue"
        elif stem.endswith("_seg"):
            stem, field = stem[:-4], "seg"
        if stem == "ct":
            modality = "ct"
        elif stem == "mr" and "PA" in path.parent.name:
            modality = "mr_t1"
        elif stem == "mr" and "PC" in path.parent.name:
            modality = "mr_t2"
        else:
            continue
        scans[modality].setdefault(path.parent, {})[field] = str(path)

    configs = []
    for key, modality in MODALITIES.items():
        if key not in requested:
            continue
        rows = sorted(scans[key].items())
        configs.append(catalog_config(
            f"{prefix}_{key}",
            [files.get("img", "") for _, files in rows],
            [files.get("seg", "") for _, files in rows],
            [files.get("seg_tissue", "") for _, files in rows],
            [folder.name for folder, _ in rows],
            modality,
        ))
    return configs
