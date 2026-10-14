"""UiTM MS/NMO 2D image and lesion-mask adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from seg2med_workflow.config import resolve_path

from .common import DatasetConfig, catalog_config

REQUIRED_CONFIG_FIELDS = (("image_root", "mask_root"),)
DEFAULT_MODALITY = 7


def natural_sort_key(value: str):
    """Sort filenames with embedded numbers in human order."""
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def uitm_patient_id_from_stem(stem: str) -> str:
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def prepare_uitm_ms_nmo(images_dir: Path, masks_dir: Path):
    """Pair slices by filename stem, retaining missing counterparts as blanks."""
    def scan(root: Path) -> dict[str, Path]:
        if not root.is_dir():
            raise FileNotFoundError(f"UiTM folder does not exist: {root}")
        return {
            path.stem: path for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        }

    images_by_stem, masks_by_stem = scan(Path(images_dir)), scan(Path(masks_dir))
    stems = sorted(images_by_stem.keys() | masks_by_stem.keys(), key=natural_sort_key)
    images = [str(images_by_stem[stem]) if stem in images_by_stem else "" for stem in stems]
    masks = [str(masks_by_stem[stem]) if stem in masks_by_stem else "" for stem in stems]
    ids = [uitm_patient_id_from_stem(stem) for stem in stems]
    return images, masks, masks.copy(), ids


def build(entry: dict[str, Any], data_root: Path) -> list[DatasetConfig]:
    image_root = resolve_path(entry.get("image_root"), data_root, required=True)
    mask_root = resolve_path(entry.get("mask_root"), data_root, required=True)
    assert image_root is not None and mask_root is not None
    images, masks, tissues, patient_ids = prepare_uitm_ms_nmo(image_root, mask_root)
    return [
        catalog_config(
            str(entry.get("name", "uitm_ms_nmo_mri")),
            images,
            masks,
            tissues,
            patient_ids,
            int(entry.get("modality", DEFAULT_MODALITY)),
        )
    ]
