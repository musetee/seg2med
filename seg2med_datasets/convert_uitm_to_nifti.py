#!/usr/bin/env python3
"""Convert UiTM MS/NMO 2D image slices into patient-level NIfTI volumes."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import nibabel as nib
import numpy as np

try:
    from .datasets.uitm import natural_sort_key, uitm_patient_id_from_stem
except ImportError:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from seg2med_datasets.datasets.uitm import natural_sort_key, uitm_patient_id_from_stem


DEFAULT_DATA_ROOT = Path(os.environ.get("SEG2MED_DATA_ROOT", "/absolute/path/to/seg2med_datasets"))
DEFAULT_UITM_ROOT = DEFAULT_DATA_ROOT / "UiTM MS NMO Dataset"
DEFAULT_IMAGES_DIR = DEFAULT_UITM_ROOT / "UITM_MS_NMO_Images"
DEFAULT_MASKS_DIR = DEFAULT_UITM_ROOT / "UITM_MS_NMO_GroundTruth_Masks"
DEFAULT_OUTPUT_DIR = DEFAULT_UITM_ROOT / "nifti_volumes"
RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def iter_raster_files(folder: Path, extensions: set[str] = RASTER_EXTENSIONS) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    return sorted(
        [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in extensions],
        key=lambda path: natural_sort_key(path.stem),
    )


def group_by_patient(files: Iterable[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        grouped[uitm_patient_id_from_stem(path.stem)].append(path)
    return {
        patient_id: sorted(paths, key=lambda path: natural_sort_key(path.stem))
        for patient_id, paths in sorted(grouped.items(), key=lambda item: natural_sort_key(item[0]))
    }


def read_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def stack_slices(paths: Sequence[Path], *, dtype: np.dtype | None = None) -> np.ndarray:
    if not paths:
        raise ValueError("Cannot create a volume from an empty slice list.")

    slices = [read_grayscale(path) for path in paths]
    shape = slices[0].shape
    mismatched = [(path.name, image.shape) for path, image in zip(paths, slices) if image.shape != shape]
    if mismatched:
        preview = ", ".join(f"{name}:{size}" for name, size in mismatched[:5])
        raise ValueError(f"All slices for one patient must have the same shape. Mismatched: {preview}")

    volume = np.stack(slices, axis=-1)
    if dtype is not None:
        volume = volume.astype(dtype)
    return volume


def save_nifti(volume: np.ndarray, output_path: Path, *, spacing: tuple[float, float, float]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    volume = np.rot90(volume, k=1, axes=(0, 1))
    affine = np.diag([spacing[1], spacing[0], spacing[2], 1.0])
    image = nib.Nifti1Image(volume, affine)
    nib.save(image, str(output_path))


def convert_uitm_to_nifti(
    *,
    images_dir: Path = DEFAULT_IMAGES_DIR,
    masks_dir: Path = DEFAULT_MASKS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    save_masks: bool = True,
    overwrite: bool = False,
) -> list[Path]:
    image_groups = group_by_patient(iter_raster_files(images_dir))
    mask_groups = group_by_patient(iter_raster_files(masks_dir)) if save_masks else {}

    if save_masks:
        missing_masks = sorted(set(image_groups) - set(mask_groups), key=natural_sort_key)
        missing_images = sorted(set(mask_groups) - set(image_groups), key=natural_sort_key)
        if missing_masks or missing_images:
            details = []
            if missing_masks:
                details.append(f"patients without masks: {missing_masks}")
            if missing_images:
                details.append(f"mask patients without images: {missing_images}")
            raise ValueError("UiTM patient image/mask mismatch; " + "; ".join(details))

    image_output_dir = output_dir / "images"
    mask_output_dir = output_dir / "lesion_masks"
    written_images: list[Path] = []

    for patient_id, image_paths in image_groups.items():
        image_output_path = image_output_dir / f"{patient_id}.nii.gz"
        if overwrite or not image_output_path.exists():
            image_volume = stack_slices(image_paths, dtype=np.uint8)
            save_nifti(image_volume, image_output_path, spacing=spacing)
        written_images.append(image_output_path)

        if save_masks:
            mask_paths = mask_groups[patient_id]
            image_stems = [path.stem for path in image_paths]
            mask_stems = [path.stem for path in mask_paths]
            if image_stems != mask_stems:
                raise ValueError(
                    f"Slice mismatch for patient {patient_id}: "
                    f"image stems {image_stems[:5]} vs mask stems {mask_stems[:5]}"
                )
            mask_output_path = mask_output_dir / f"{patient_id}_lesion.nii.gz"
            if overwrite or not mask_output_path.exists():
                mask_volume = (stack_slices(mask_paths) > 0).astype(np.uint8)
                save_nifti(mask_volume, mask_output_path, spacing=spacing)

    file_list_path = output_dir / "uitm_ms_nmo_nifti_images.txt"
    file_list_path.parent.mkdir(parents=True, exist_ok=True)
    file_list_path.write_text("\n".join(str(path) for path in written_images) + "\n", encoding="utf-8")
    return written_images


def parse_spacing(value: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Spacing must have three comma-separated values, e.g. 1,1,1")
    return parts[0], parts[1], parts[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert UiTM MS/NMO 2D slices to NIfTI volumes.")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--masks-dir", type=Path, default=DEFAULT_MASKS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--spacing", type=parse_spacing, default=(1.0, 1.0, 1.0), help="Voxel spacing as x,y,z.")
    parser.add_argument("--no-masks", action="store_true", help="Only convert source images.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing NIfTI files.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outputs = convert_uitm_to_nifti(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
        spacing=args.spacing,
        save_masks=not args.no_masks,
        overwrite=args.overwrite,
    )
    print(f"Wrote/listed {len(outputs)} image volumes")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

