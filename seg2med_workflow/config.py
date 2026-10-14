"""Small, dependency-light helpers shared by the stage entry points."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when a workflow YAML file is missing required configuration."""


def load_yaml(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load a YAML mapping and return it with the config file directory."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("PyYAML is required to read seg2med stage configs.") from exc

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"The YAML root must be a mapping: {config_path}")
    return data, config_path.parent


def require_mapping(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' must be a mapping.")
    return value


def resolve_path(
    value: str | Path | None,
    base_dir: str | Path,
    *,
    required: bool = False,
) -> Path | None:
    """Expand environment variables and resolve a path against ``base_dir``."""
    if value is None or str(value).strip() == "":
        if required:
            raise ConfigError("A required path is missing.")
        return None
    raw_value = str(value)
    expanded = os.path.expandvars(os.path.expanduser(raw_value))
    unresolved = re.findall(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)|%([^%]+)%", expanded)
    if unresolved:
        names = sorted({next(part for part in match if part) for match in unresolved})
        raise ConfigError(f"Environment variable(s) not set in path '{raw_value}': {', '.join(names)}")
    path = Path(expanded)
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path.resolve()


def as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    raise ConfigError(f"Expected a boolean value, got {value!r}.")
