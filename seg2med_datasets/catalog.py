"""Lightweight catalog assembly for dataset-preparation stage outputs."""

from __future__ import annotations

import logging
from itertools import zip_longest
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import SCHEMA, MODALITY_NAMES

logger = logging.getLogger(__name__)


def _clean(value: Any) -> str:
    """Return a CSV-safe string, using an empty string for a missing value."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _alignment_key(value: Any) -> tuple[str, str] | None:
    """Identify image and mask paths that describe the same scan."""
    cleaned = _clean(value)
    if not cleaned:
        return None

    path = Path(cleaned)
    name = path.name.casefold()
    for extension in (".nii.gz", ".nii", ".nrrd", ".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        if name.endswith(extension):
            name = name[: -len(extension)]
            break
    for suffix in ("_seg_tissue", "_seg", "_mask", "_image", "_img"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    parent = path.parent.as_posix().casefold()
    return parent, name


def _infer_patient_id(*paths: Any) -> str:
    for value in paths:
        cleaned = _clean(value)
        if not cleaned:
            continue
        path = Path(cleaned)
        if path.parent.name:
            return path.parent.name
        key = _alignment_key(cleaned)
        if key is not None:
            return key[1]
    return ""


def _aligned_values(config: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """Align catalog paths, retaining rows whose image or mask is absent.

    Equal-length lists keep their established positional meaning. When scanners
    return unequal lists, known image/mask filename suffixes are used to avoid
    shifting every row after a missing file. If filenames cannot be related,
    the lists are padded positionally as a conservative fallback.
    """
    images = list(config[SCHEMA.CFG_IMAGES])
    segs = list(config[SCHEMA.CFG_SEGS])
    tissues = list(config[SCHEMA.CFG_SEG_TISSUES])
    patient_ids = list(config[SCHEMA.CFG_PIDS])
    values = [images, segs, tissues, patient_ids]
    if len({len(items) for items in values}) == 1:
        return [tuple(_clean(value) for value in item) for item in zip(*values)]

    logger.warning(
        "Dataset '%s' has unaligned image/mask/id lists (images=%d, segs=%d, "
        "seg_tissues=%d, ids=%d); missing entries will be empty.",
        config[SCHEMA.CFG_NAME],
        len(images),
        len(segs),
        len(tissues),
        len(patient_ids),
    )

    path_lists = [images, segs, tissues]
    keyed_lists: list[dict[tuple[str, str], str]] = []
    for items in path_lists:
        keyed: dict[tuple[str, str], str] = {}
        for value in items:
            key = _alignment_key(value)
            if key is None or key in keyed:
                continue
            keyed[key] = _clean(value)
        keyed_lists.append(keyed)

    key_sets = [set(items) for items in keyed_lists if items]
    has_path_matches = any(
        left & right
        for index, left in enumerate(key_sets)
        for right in key_sets[index + 1 :]
    )
    unique_keys = all(
        len(keyed) == sum(_alignment_key(value) is not None for value in source)
        for keyed, source in zip(keyed_lists, path_lists)
    )

    if has_path_matches and unique_keys:
        ordered_keys: list[tuple[str, str]] = []
        # The longest list normally contains the complete scan order. This also
        # preserves ID order when, for example, one image is missing but masks
        # and patient IDs still contain every scan.
        for keyed in sorted(keyed_lists, key=len, reverse=True):
            for key in keyed:
                if key not in ordered_keys:
                    ordered_keys.append(key)

        ids_by_image_key = {}
        if len(patient_ids) == len(images):
            ids_by_image_key = {
                key: _clean(patient_id)
                for image, patient_id in zip(images, patient_ids)
                if (key := _alignment_key(image)) is not None
            }
        aligned: list[tuple[str, str, str, str]] = []
        for index, key in enumerate(ordered_keys):
            image = keyed_lists[0].get(key, "")
            seg = keyed_lists[1].get(key, "")
            tissue = keyed_lists[2].get(key, "")
            patient_id = ids_by_image_key.get(key, "")
            if not patient_id and len(patient_ids) == len(ordered_keys):
                patient_id = _clean(patient_ids[index])
            if not patient_id:
                patient_id = _infer_patient_id(image, seg, tissue)
            aligned.append((image, seg, tissue, patient_id))
        return aligned

    aligned = []
    for image, seg, tissue, patient_id in zip_longest(*values, fillvalue=""):
        cleaned = (_clean(image), _clean(seg), _clean(tissue), _clean(patient_id))
        if not cleaned[3]:
            cleaned = (*cleaned[:3], _infer_patient_id(*cleaned[:3]))
        aligned.append(cleaned)
    return aligned


def build_dataframe(dataset_configs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    row_policies = []
    for config in dataset_configs:
        modality = int(config[SCHEMA.CFG_MODALITY])
        for image, segmentation, tissue, patient_id in _aligned_values(config):
            row_policies.append({key: value for key, value in config.items() if key.startswith("split_")})
            rows.append(
                {
                    SCHEMA.DATASET: str(config[SCHEMA.CFG_NAME]),
                    SCHEMA.ID: str(patient_id).strip(),
                    SCHEMA.SEG: str(segmentation).strip(),
                    SCHEMA.SEG_TISSUE: str(tissue).strip(),
                    SCHEMA.IMG: str(image).strip(),
                    SCHEMA.MODALITY_ID: modality,
                    SCHEMA.MODALITY_NAME: MODALITY_NAMES.get(modality, f"unknown_{modality}"),
                    SCHEMA.AORTA_DISS: 0,
                }
            )
    if not rows:
        raise ValueError("Dataset scanning produced no catalog rows.")
    frame = pd.DataFrame(rows, columns=list(SCHEMA.ALL))
    frame.attrs["split_policies"] = row_policies
    return frame


def validate_dataframe(frame: pd.DataFrame, *, check_paths: bool) -> None:
    required = list(SCHEMA.ALL)
    missing_columns = set(required) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Catalog is missing columns: {sorted(missing_columns)}")
    metadata_columns = [
        SCHEMA.DATASET,
        SCHEMA.MODALITY_ID,
        SCHEMA.MODALITY_NAME,
        SCHEMA.AORTA_DISS,
    ]
    missing_metadata = frame[metadata_columns].apply(lambda column: column.map(lambda value: not _clean(value)))
    if missing_metadata.any().any():
        raise ValueError("Catalog contains missing required metadata values.")
    path_columns = [SCHEMA.IMG, SCHEMA.SEG, SCHEMA.SEG_TISSUE]
    cleaned_paths = frame[path_columns].apply(lambda column: column.map(_clean))
    if cleaned_paths.eq("").all(axis=1).any():
        raise ValueError("Catalog contains a row without an image or segmentation path.")
    has_image = frame[SCHEMA.IMG].map(lambda value: bool(_clean(value)))
    duplicates = has_image & frame.duplicated(subset=[SCHEMA.DATASET, SCHEMA.ID, SCHEMA.IMG], keep=False)
    if duplicates.any():
        raise ValueError(f"Catalog contains duplicate patient/image rows:\n{frame.loc[duplicates].head()}")
    if check_paths:
        for column in (SCHEMA.IMG, SCHEMA.SEG, SCHEMA.SEG_TISSUE):
            present = frame[column].map(_clean)
            missing = frame.loc[present.ne("") & ~present.map(lambda value: Path(value).is_file()), column]
            if not missing.empty:
                raise FileNotFoundError(f"Catalog path in '{column}' does not exist: {missing.iloc[0]}")


def _write(frame: pd.DataFrame, csv_path: Path, parquet_path: Path | None) -> None:
    frame.to_csv(csv_path, index=False)
    if parquet_path is not None:
        frame.to_parquet(parquet_path, index=False)


def build_and_save_catalog(
    dataset_configs: list[dict[str, Any]],
    *,
    output_dir: str | Path,
    out_csv: str,
    out_parquet: str | None,
    check_paths: bool,
    split_prefix: str | None,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    included_modalities: Any = (0, 1, 2, 3, 4),
    patient_mapping: str | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = build_dataframe(dataset_configs)
    validate_dataframe(frame, check_paths=check_paths)
    split_result = {}
    train = validation = test = None
    if split_prefix:
        from .splitting import build_manifests

        mapping_path = output / (patient_mapping or f"{split_prefix}_patient_splits.csv")
        split_result = build_manifests(
            frame, dataset_configs, ratios=(train_ratio, validation_ratio, test_ratio),
            seed=seed, mapping_path=mapping_path, included_modalities=included_modalities,
        )
        train, validation, test = (split_result["manifests"][name] for name in ("train", "validation", "test"))
        for name, manifest in split_result["manifests"].items():
            split_parquet = output / f"{split_prefix}_{name}.parquet" if out_parquet else None
            _write(manifest, output / f"{split_prefix}_{name}.csv", split_parquet)
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        split_result["patient_mapping"].to_csv(mapping_path, index=False)
        split_result["identity_audit"].to_csv(output / f"{split_prefix}_identity_audit.csv", index=False)
        split_result["excluded_df"].to_csv(output / f"{split_prefix}_excluded.csv", index=False)
        split_result["patient_mapping_path"] = mapping_path
    parquet_path = output / out_parquet if out_parquet else None
    _write(frame, output / out_csv, parquet_path)
    modality_columns = [SCHEMA.MODALITY_ID, SCHEMA.MODALITY_NAME]
    modality_stats = frame.groupby(modality_columns).size().to_frame("num_items")
    # IDs are dataset-local, and one case may have multiple scans or slices.
    modality_stats["unique_cases"] = (
        frame.drop_duplicates(subset=[*modality_columns, SCHEMA.DATASET, SCHEMA.ID])
        .groupby(modality_columns).size()
    )
    if split_result:
        audit = split_result["identity_audit"]
        modality_stats["unique_cases"] = (
            audit.loc[audit.identity_issue.eq("")]
            .drop_duplicates(subset=[*modality_columns, "source", "patient_id"])
            .groupby(modality_columns).size()
        ).reindex(modality_stats.index, fill_value=0)
    stats = {
        "by_dataset": frame.groupby(SCHEMA.DATASET)[SCHEMA.ID].nunique().to_frame("unique_patients"),
        "by_modality": modality_stats,
    }
    return {"df": frame, "train_df": train, "validation_df": validation, "test_df": test,
            "stats": stats, **split_result}
