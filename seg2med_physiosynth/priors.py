"""Catalog-level IO for physics-informed prior generation."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from seg2med_workflow.config import ConfigError, as_bool, load_yaml, resolve_path

from .simulator import MODALITY_SPECS, simulate_prior

PHYSIOSYNTH_ROOT = Path(__file__).resolve().parent
DEFAULT_CT_PARAMETERS = PHYSIOSYNTH_ROOT / "params_ct.csv"
DEFAULT_MR_PARAMETERS = PHYSIOSYNTH_ROOT / "params_mr.csv"


def _output_name(path: Path, suffix: str) -> str:
    if path.name.lower().endswith(".nii.gz"):
        return f"{path.name[:-7]}{suffix}.nii.gz"
    return f"{path.stem}{suffix}{path.suffix}"


def _csv_path(path: Path, root: Path, *, absolute: bool) -> str:
    return str(path.resolve()) if absolute else os.path.relpath(path, start=root)


def _prior_paths(image_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    raster = image_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    prior_name = f"{image_path.stem}_prior_image.png" if raster else _output_name(image_path, "_prior_image")
    semantic_name = f"{image_path.stem}_prior_semantic.png" if raster else _output_name(image_path, "_prior_semantic")
    return output_dir / image_path.name, output_dir / prior_name, output_dir / semantic_name


def _prepare_raster(image_path: Path, seg_path: Path, output_dir: Path, *, overwrite: bool = True) -> tuple[Path, Path, Path]:
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    lesion = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
    if image is None or lesion is None:
        raise FileNotFoundError(f"Could not read raster image or mask: {image_path}, {seg_path}")
    binary = (lesion > 0).astype(np.uint8)
    prior = (image.astype(np.float32) * binary).clip(0, 255).astype(np.uint8)
    semantic = binary * 255
    target_path, prior_path, semantic_path = _prior_paths(image_path, output_dir)
    if overwrite or not target_path.is_file():
        shutil.copy2(image_path, target_path)
    for path, array in ((prior_path, prior), (semantic_path, semantic)):
        if (overwrite or not path.is_file()) and not cv2.imwrite(str(path), array):
            raise IOError(f"Could not write raster prior: {path}")
    return target_path, prior_path, semantic_path


def _prepare_volume(
    image_path: Path,
    seg_path: Path,
    tissue_path: Path,
    output_dir: Path,
    modality: int,
    *,
    ct_parameters: Path,
    mr_parameters: Path,
    min_contour_area: float,
    threshold: float | None,
    overwrite: bool = True,
    output_paths: tuple[Path, Path, Path] | None = None,
) -> tuple[Path, Path, Path]:
    import nibabel as nib

    image_nii = nib.load(str(image_path))
    image = image_nii.get_fdata()
    organs = nib.load(str(seg_path)).get_fdata()
    tissues = nib.load(str(tissue_path)).get_fdata()
    prior, semantic, contour = simulate_prior(
        image,
        organs,
        tissues,
        modality,
        ct_parameters=ct_parameters,
        mr_parameters=mr_parameters,
        body_threshold=threshold,
        min_contour_area=min_contour_area,
    )
    spec = MODALITY_SPECS[modality]
    processed_target = image * contour + spec.background * (1 - contour)
    target_path, prior_path, semantic_path = output_paths or _prior_paths(image_path, output_dir)
    contour_path = output_dir / _output_name(image_path, "_contour")
    def save(array: np.ndarray, path: Path, dtype: np.dtype) -> None:
        if not overwrite and path.is_file():
            return
        header = image_nii.header.copy()
        header.set_data_dtype(dtype)
        nib.save(nib.Nifti1Image(array.astype(dtype), image_nii.affine, header), str(path))

    save(processed_target, target_path, image_nii.get_data_dtype())
    save(prior, prior_path, np.dtype(np.float32))
    save(semantic, semantic_path, np.dtype(np.int16))
    save(contour, contour_path, np.dtype(np.uint8))
    for source in (seg_path, tissue_path):
        destination = output_dir / source.name
        if source.resolve() == destination.resolve():
            continue
        if overwrite or not destination.is_file():
            shutil.copy2(source, destination)
    return target_path, prior_path, semantic_path


def _modalities(value: Any) -> set[int]:
    if value in (None, "all", "*"):
        return set(MODALITY_SPECS)
    if not isinstance(value, list):
        raise ConfigError("'modalities' must be 'all' or a list of modality ids.")
    return {int(item) for item in value}


def create_priors_from_config(config_path: str | Path, *, dry_run: bool = False,
                              overwrite: bool | None = None, csv_only: bool | None = None,
                              absolute_paths: bool | None = None) -> dict[str, Any]:
    """Export prior catalogs; csv_only records planned paths without image IO.

    Referenced outputs need not exist in CSV-only mode. dry_run takes precedence
    and suppresses even CSV writes.
    """
    import pandas as pd

    config, config_dir = load_yaml(config_path)
    overwrite = as_bool(config.get("overwrite"), default=False) if overwrite is None else as_bool(overwrite)
    csv_only = as_bool(config.get("csv_only"), default=False) if csv_only is None else as_bool(csv_only)
    absolute_paths = (as_bool(config.get("absolute_paths"), default=False)
                      if absolute_paths is None else as_bool(absolute_paths))
    combined_csv = resolve_path(config.get("combined_csv", "seg2med_all.csv"), config_dir, required=True)
    assert combined_csv is not None
    data_root = resolve_path(config.get("data_root", "."), config_dir, required=True)
    assert data_root is not None
    parameters = config.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ConfigError("'parameters' must be a mapping.")
    ct_parameters = resolve_path(parameters.get("ct", DEFAULT_CT_PARAMETERS), config_dir, required=True)
    mr_parameters = resolve_path(parameters.get("mr", DEFAULT_MR_PARAMETERS), config_dir, required=True)
    assert ct_parameters is not None and mr_parameters is not None
    allowed_modalities = _modalities(config.get("modalities", "all"))
    min_contour_area = float(config.get("min_contour_area", 10000))
    thresholds = config.get("body_thresholds", {})
    if not isinstance(thresholds, dict):
        raise ConfigError("'body_thresholds' must be a mapping of modality id to value.")

    splits = config.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise ConfigError("'splits' must be a non-empty mapping.")
    if "combined" in splits:
        raise ConfigError("'combined' is reserved for the merged train/test CSV result.")
    for name, split in splits.items():
        if not isinstance(split, dict):
            raise ConfigError(f"Split '{name}' must be a mapping.")
        for value in (split.get("catalog"), split.get("output_csv", f"{name}.csv")):
            if resolve_path(value, config_dir) == combined_csv:
                raise ConfigError("combined_csv must differ from all input catalogs and split output CSVs.")
    result: dict[str, Any] = {}
    combined_frames = []
    combined_rows = 0
    for split_name, split in splits.items():
        if not isinstance(split, dict):
            raise ConfigError(f"Split '{split_name}' must be a mapping.")
        catalog_path = resolve_path(split.get("catalog"), config_dir, required=True)
        output_dir = resolve_path(split.get("output_dir", f"generated/{split_name}"), data_root, required=True)
        output_csv = resolve_path(split.get("output_csv", f"{split_name}.csv"), config_dir, required=True)
        assert catalog_path is not None and output_dir is not None and output_csv is not None
        frame = pd.read_csv(catalog_path)
        required_columns = {"id", "img", "modality"}
        if not csv_only:
            required_columns.update({"seg", "seg_tissue"})
        missing = required_columns - set(frame.columns)
        if missing:
            raise ConfigError(f"Catalog {catalog_path} is missing columns: {sorted(missing)}")
        selected = frame[frame["modality"].astype(int).isin(allowed_modalities)]
        combined_rows += len(selected)
        generated = reused = 0
        rows: list[dict[str, Any]] = []
        for row in selected.to_dict(orient="records"):
            image_path = resolve_path(row["img"], data_root, required=True)
            if pd.isna(row["img"]):
                raise ConfigError("An image path is required to determine prior output filenames.")
            assert image_path is not None
            patient_dir = output_dir / str(row["id"])
            target, prior, semantic = _prior_paths(image_path, patient_dir)
            raster = image_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            complete = True
            if not csv_only:
                seg_path = resolve_path(row["seg"], data_root, required=True)
                tissue_path = resolve_path(row["seg_tissue"], data_root, required=True)
                assert seg_path is not None and tissue_path is not None
                expected = [target, prior, semantic]
                if not raster:
                    expected += [
                        patient_dir / _output_name(image_path, "_contour"),
                        patient_dir / seg_path.name,
                        patient_dir / tissue_path.name,
                    ]
                complete = not overwrite and all(path.is_file() for path in expected)
                if complete:
                    reused += 1
                else:
                    generated += 1
            if dry_run:
                continue
            if not complete:
                patient_dir.mkdir(parents=True, exist_ok=True)
            if not complete and raster:
                target, prior, semantic = _prepare_raster(image_path, seg_path, patient_dir, overwrite=overwrite)
            elif not complete:
                modality = int(row["modality"])
                target, prior, semantic = _prepare_volume(
                    image_path,
                    seg_path,
                    tissue_path,
                    patient_dir,
                    modality,
                    ct_parameters=ct_parameters,
                    mr_parameters=mr_parameters,
                    min_contour_area=min_contour_area,
                    threshold=(
                        float(thresholds[str(modality)])
                        if str(modality) in thresholds
                        else float(thresholds[modality]) if modality in thresholds else None
                    ),
                    overwrite=overwrite,
                )
            rows.append(
                {
                    "id": row["id"],
                    "aorta_diss": int(row.get("aorta_diss", 0)),
                    "prior": _csv_path(prior, data_root, absolute=absolute_paths),
                    "img": _csv_path(target, data_root, absolute=absolute_paths),
                    "modality": int(row["modality"]),
                    "prior_semantic": _csv_path(semantic, data_root, absolute=absolute_paths),
                }
            )
        if dry_run:
            result[str(split_name)] = {
                "rows": len(selected), "output_dir": output_dir, "output_csv": output_csv,
                "generated": generated, "reused": reused,
                "csv_only": csv_only,
            }
            continue
        output_frame = pd.DataFrame(rows, columns=["id", "aorta_diss", "prior", "img", "modality", "prior_semantic"])
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        output_frame.to_csv(output_csv, index=False)
        if not csv_only and as_bool(split.get("write_excel"), default=False):
            output_frame.to_excel(output_csv.with_suffix(".xlsx"), index=False)
        result[str(split_name)] = output_frame
        output_frame.attrs.update(generated=generated, reused=reused, csv_only=csv_only)
        combined_frames.append(output_frame.assign(split=str(split_name)))
    if dry_run:
        result["combined"] = {"rows": combined_rows, "output_csv": combined_csv, "csv_only": csv_only}
    else:
        combined_frame = pd.concat(combined_frames, ignore_index=True)
        combined_frame.attrs.clear()
        combined_frame.attrs["csv_only"] = csv_only
        combined_csv.parent.mkdir(parents=True, exist_ok=True)
        combined_frame.to_csv(combined_csv, index=False)
        result["combined"] = combined_frame
    return result
