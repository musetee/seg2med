"""Dataset-specific catalog adapters.

Each supported dataset lives in its own module.  Add a new adapter there and
register it in :mod:`seg2med_datasets.datasets.registry`.
"""

from .registry import (
    available_dataset_kinds,
    build_dataset_configs,
    describe_dataset_modalities,
    describe_dataset_requirements,
)

__all__ = [
    "available_dataset_kinds", "build_dataset_configs",
    "describe_dataset_modalities", "describe_dataset_requirements",
]
