"""Registry for independently maintained dataset adapters."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from seg2med_workflow.config import ConfigError

from .common import DatasetConfig
from ..schema import MODALITY_NAMES

DatasetBuilder = Callable[[dict[str, Any], Path], list[DatasetConfig]]


def _adapters() -> dict[str, ModuleType]:
    from . import anika, anish, csv_dataset, generic, synthrad, uitm

    return {
        "anika": anika,
        "anish": anish,
        "csv": csv_dataset,
        "generic": generic,
        "synthrad": synthrad,
        "trial": synthrad,
        "umm": csv_dataset,
        "uitm_ms_nmo": uitm,
    }


def _builders() -> dict[str, DatasetBuilder]:
    return {kind: adapter.build for kind, adapter in _adapters().items()}


def describe_dataset_modalities() -> dict[str, str]:
    """Describe YAML modality choices without scanning any dataset files."""
    descriptions = {}
    for kind, adapter in sorted(_adapters().items()):
        if hasattr(adapter, "MODALITIES"):
            choices = ", ".join(f"{name} (ID {value})" for name, value in adapter.MODALITIES.items())
            descriptions[kind] = f"modalities: {choices}"
        else:
            default = adapter.DEFAULT_MODALITY
            descriptions[kind] = f"modality: configurable integer; default {default} ({MODALITY_NAMES.get(default, 'unknown')})"
            if hasattr(adapter, "ROOT_MODALITIES"):
                choices = ", ".join(f"{name} (ID {value})" for name, value in adapter.ROOT_MODALITIES.items())
                descriptions[kind] = f"root modalities: {choices}; CSV {descriptions[kind]}"
    return descriptions


def describe_dataset_requirements() -> dict[str, str]:
    """Describe required per-entry YAML fields; groups are alternatives.

    All fields within a group are required together. Shared adapters supply
    the same requirements for aliases, without scanning any dataset files.
    """
    return {
        kind: " OR ".join(" + ".join(group) for group in adapter.REQUIRED_CONFIG_FIELDS)
        for kind, adapter in sorted(_adapters().items())
    }


def available_dataset_kinds() -> list[str]:
    return sorted(_builders())


def build_dataset_configs(entry: dict[str, Any], data_root: Path) -> list[DatasetConfig]:
    kind = str(entry.get("kind", "generic")).strip().lower()
    try:
        builder = _builders()[kind]
    except KeyError as exc:
        raise ConfigError(
            f"Unsupported dataset kind '{kind}'. Choose from: {', '.join(available_dataset_kinds())}"
        ) from exc
    return builder(entry, data_root)
