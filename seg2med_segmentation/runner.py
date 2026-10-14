"""Configuration and execution for the segmentation stage."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Iterable

from seg2med_workflow.config import ConfigError, as_bool, load_yaml, resolve_path

NIFTI_EXTENSIONS = (".nii", ".nii.gz", ".nrrd")
TISSUE_TASKS = {"tissue_types", "tissue_types_mr"}


def _read_path_list(path: Path, base_dir: Path) -> list[Path]:
    items: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            resolved = resolve_path(value, base_dir, required=True)
            assert resolved is not None
            items.append(resolved)
    return items


def _from_catalog(source: dict[str, Any], base_dir: Path) -> list[Path]:
    import pandas as pd

    catalog = resolve_path(source.get("path"), base_dir, required=True)
    assert catalog is not None
    frame = pd.read_csv(catalog)
    image_column = str(source.get("image_column", "img"))
    if image_column not in frame.columns:
        raise ConfigError(f"Catalog has no '{image_column}' column: {catalog}")
    selected_datasets = source.get("datasets")
    if selected_datasets:
        dataset_column = str(source.get("dataset_column", "dataset"))
        if dataset_column not in frame.columns:
            raise ConfigError(f"Catalog has no '{dataset_column}' column: {catalog}")
        frame = frame[frame[dataset_column].astype(str).isin({str(value) for value in selected_datasets})]
    modalities = source.get("modalities")
    if modalities:
        modality_column = str(source.get("modality_column", "modality"))
        if modality_column not in frame.columns:
            raise ConfigError(f"Catalog has no '{modality_column}' column: {catalog}")
        frame = frame[frame[modality_column].astype(str).isin({str(value) for value in modalities})]
    paths: list[Path] = []
    for value in frame[image_column].tolist():
        if value is None or pd.isna(value) or not str(value).strip():
            continue
        resolved = resolve_path(str(value).strip(), catalog.parent, required=True)
        assert resolved is not None
        paths.append(resolved)
    return paths


def collect_inputs(source: dict[str, Any], base_dir: Path) -> list[Path]:
    """Collect image paths from a catalog, glob, text list, or inline list."""
    source_type = str(source.get("type", "catalog")).lower()
    if source_type == "catalog":
        paths = _from_catalog(source, base_dir)
    elif source_type == "glob":
        root = resolve_path(source.get("root", "."), base_dir, required=True)
        assert root is not None
        pattern = str(source.get("pattern", "**/*.nii.gz"))
        paths = [Path(value).resolve() for value in glob.glob(str(root / pattern), recursive=True)]
    elif source_type == "list":
        list_path = resolve_path(source.get("path"), base_dir, required=True)
        assert list_path is not None
        paths = _read_path_list(list_path, list_path.parent)
    elif source_type == "paths":
        values = source.get("paths", [])
        if not isinstance(values, list):
            raise ConfigError("source.paths must be a list.")
        paths = []
        for value in values:
            resolved = resolve_path(value, base_dir, required=True)
            assert resolved is not None
            paths.append(resolved)
    elif source_type == "legacy_dataset":
        raise ConfigError("source.type 'legacy_dataset' has been removed. Use the stage-1 catalog or a glob source.")
    else:
        raise ConfigError("source.type must be catalog, glob, list, or paths.")

    unique = sorted({path for path in paths}, key=lambda path: path.as_posix().lower())
    limit = source.get("limit")
    return unique[: int(limit)] if limit is not None else unique


def _strip_image_extension(name: str) -> str:
    return name[:-7] if name.lower().endswith(".nii.gz") else Path(name).stem


def output_path(input_path: Path, task: str, output_dir: Path | None, multi_label: bool) -> Path:
    suffix = "_seg_tissue" if task in TISSUE_TASKS else "_seg"
    parent = output_dir if output_dir is not None else input_path.parent
    stem = _strip_image_extension(input_path.name)
    if multi_label:
        return parent / f"{stem}{suffix}.nii.gz"
    return parent / f"{stem}{suffix}"


def _load_input(path: Path):
    if path.name.lower().endswith((".nii", ".nii.gz")):
        import nibabel as nib

        return nib.load(str(path))
    if path.suffix.lower() == ".nrrd":
        import nibabel as nib
        import nrrd
        import numpy as np

        data, header = nrrd.read(str(path))
        directions = header.get("space directions")
        affine = np.eye(4)
        if directions is not None:
            affine[:3, :3] = np.asarray(directions)
        affine[:3, 3] = np.asarray(header.get("space origin", [0, 0, 0]))
        return nib.Nifti1Image(data, affine)
    raise ValueError(f"Unsupported segmentation input: {path}")


def _task_names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return value
    raise ConfigError("'tasks' must be a task name or a non-empty list of task names.")


def segment_volume(image_path: Path, target: Path, *, task: str,
                   device: str = "gpu", multi_label: bool = True, fast: bool = False) -> None:
    """Segment one volume; shared by batch and single-case preparation.

    Import the optional backend only when work is actually needed. CUDA names
    are translated to the names accepted by TotalSegmentator's Python API.
    """
    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError as exc:
        raise ImportError("TotalSegmentator is required for the segmentation stage.") from exc
    device = str(device).lower()
    if device == "cuda" or device.startswith("cuda:"):
        device = device.replace("cuda", "gpu", 1)
    target.parent.mkdir(parents=True, exist_ok=True)
    totalsegmentator(input=_load_input(image_path), output=str(target), task=task,
                    fast=fast, ml=multi_label, device=device)


def run_from_config(config_path: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
    config, config_dir = load_yaml(config_path)
    source = config.get("source")
    if not isinstance(source, dict):
        raise ConfigError("'source' must be a mapping.")
    inputs = collect_inputs(source, config_dir)
    if not inputs:
        raise ConfigError("The segmentation source resolved to zero input images.")
    missing = [path for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Segmentation input does not exist: {missing[0]}")

    tasks = _task_names(config.get("tasks", "total"))
    output_dir = resolve_path(config.get("output_dir"), config_dir)
    if output_dir is not None and not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    device = str(config.get("device", "gpu"))
    multi_label = as_bool(config.get("multi_label"), default=True)
    fast = as_bool(config.get("fast"), default=False)
    overwrite = as_bool(config.get("overwrite"), default=False)
    continue_on_error = as_bool(config.get("continue_on_error"), default=False)
    planned = [(path, task, output_path(path, task, output_dir, multi_label)) for task in tasks for path in inputs]
    result: dict[str, Any] = {"planned": planned, "created": [], "skipped": [], "failed": []}
    if dry_run:
        return result

    for image_path, task, target in planned:
        if target.exists() and not overwrite:
            result["skipped"].append(target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            segment_volume(image_path, target, task=task, fast=fast,
                           multi_label=multi_label, device=device)
            result["created"].append(target)
        except Exception as exc:
            result["failed"].append((image_path, task, exc))
            if not continue_on_error:
                raise
    return result
