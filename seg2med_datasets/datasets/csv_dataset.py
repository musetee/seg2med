"""Adapter for datasets described by an existing CSV table."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from seg2med_workflow.config import ConfigError, resolve_path

from .common import DatasetConfig, catalog_config, with_nifti_suffix

REQUIRED_CONFIG_FIELDS = (("csv",),)
DEFAULT_MODALITY = 0


def build(entry: dict[str, Any], data_root: Path) -> list[DatasetConfig]:
    import pandas as pd

    csv_path = resolve_path(entry.get("csv"), data_root, required=True)
    assert csv_path is not None
    frame = pd.read_csv(csv_path, keep_default_na=False)
    if frame.empty:
        raise ConfigError(f"Dataset CSV is empty: {csv_path}")
    columns = entry.get("columns", {})
    if not isinstance(columns, dict):
        raise ConfigError("CSV dataset 'columns' must be a mapping.")

    def column(logical: str, fallback: str) -> list[Any]:
        key = str(columns.get(logical, fallback))
        if key not in frame.columns:
            raise ConfigError(f"Column '{key}' was not found in {csv_path}")
        return frame[key].tolist()

    def path_value(value: Any) -> str:
        return str(resolve_path(value, data_root, required=True)) if str(value).strip() else ""

    def inferred_mask(image: str, suffix: str) -> str:
        if not image:
            return ""
        candidate = with_nifti_suffix(Path(image), suffix)
        return str(candidate) if candidate.is_file() else ""

    images = [path_value(value) for value in column("image", "img")]
    ids = column("id", "id")
    if "segmentation" in columns or "seg" in frame.columns:
        segmentations = [path_value(value) for value in column("segmentation", "seg")]
    else:
        segmentations = [inferred_mask(path, "_seg") for path in images]
    tissue_key = str(columns.get("tissue_segmentation", "seg_tissue"))
    if tissue_key in frame.columns:
        tissues = [path_value(value) for value in frame[tissue_key].tolist()]
    else:
        tissues = [inferred_mask(path, "_seg_tissue") for path in images]
    return [
        catalog_config(
            str(entry.get("name", csv_path.stem)),
            images,
            segmentations,
            tissues,
            ids,
            int(entry.get("modality", DEFAULT_MODALITY)),
        )
    ]
