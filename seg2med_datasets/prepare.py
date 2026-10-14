"""YAML-configured dataset discovery and catalog export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from seg2med_workflow.config import ConfigError, as_bool, load_yaml, resolve_path

from .datasets import build_dataset_configs


def prepare_from_config(
    config_path: str | Path, *, dry_run: bool = False, csv_only: bool = False
) -> dict[str, Any]:
    """Export catalog tables, optionally suppressing Parquet; never create priors."""
    config, config_dir = load_yaml(config_path)
    data_root = resolve_path(config.get("data_root", "."), config_dir, required=True)
    assert data_root is not None
    entries = config.get("datasets")
    if not isinstance(entries, list) or not entries:
        raise ConfigError("'datasets' must be a non-empty list.")

    dataset_configs: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ConfigError("Every datasets entry must be a mapping.")
        built = build_dataset_configs(entry, data_root)
        if not built:
            continue
        # Adapter names include modality suffixes; grouping must use the source
        # entry instead, so all sequences/scans share one patient namespace.
        kind = str(entry.get("kind", "generic")).lower()
        default_source = kind if kind in {"anika", "synthrad", "trial"} else built[0]["name"]
        source = str(entry.get("patient_group", entry.get("name", default_source)))
        if not source.strip():
            raise ConfigError("Dataset patient_group/name must not be empty.")
        for item in built:
            item.update(split_source=source, split_kind=str(entry.get("kind", "generic")).lower(),
                        split_region=entry.get("region", ""),
                        split_patient_id_regex=entry.get("split_patient_id_regex"),
                        split_patient_id_map=entry.get("split_patient_id_map", {}))
            if not isinstance(item["split_patient_id_map"], dict):
                raise ConfigError("split_patient_id_map must map scan IDs to shared patient IDs.")
        dataset_configs.extend(built)

    summary = {
        "datasets": [item["name"] for item in dataset_configs],
        "images": sum(len(item["images"]) for item in dataset_configs),
    }
    if dry_run:
        return summary

    output = config.get("output", {})
    if not isinstance(output, dict):
        raise ConfigError("'output' must be a mapping.")
    output_dir = resolve_path(output.get("directory", "generated"), config_dir, required=True)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    split = output.get("split", {})
    if not isinstance(split, dict):
        raise ConfigError("'output.split' must be a mapping.")
    split_prefix = str(split.get("prefix", "seg2med_split")) if as_bool(split.get("enabled"), default=True) else None
    parquet = None if csv_only else output.get("parquet")

    from .catalog import build_and_save_catalog

    result = build_and_save_catalog(
        dataset_configs,
        out_csv=str(output.get("catalog", "seg2med_catalog.csv")),
        out_parquet=str(parquet) if parquet else None,
        output_dir=str(output_dir),
        check_paths=as_bool(output.get("check_paths"), default=False),
        split_prefix=split_prefix,
        train_ratio=float(split.get("train_ratio", 0.7)),
        validation_ratio=float(split.get("validation_ratio", 0.15)),
        test_ratio=float(split.get("test_ratio", 0.15)),
        included_modalities=split.get("modalities", [0, 1, 2, 3, 4]),
        patient_mapping=split.get("patient_mapping"),
        seed=int(split.get("seed", 42)),
    )
    result["output_dir"] = output_dir
    return result
