"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file as a dictionary."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must contain a mapping at the top level: {config_path}")
    return data


def save_yaml_config(config: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary to a YAML config file."""
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def resolve_project_path(*parts: str) -> Path:
    """Resolve path parts relative to the repository root."""
    return _project_root().joinpath(*parts)
