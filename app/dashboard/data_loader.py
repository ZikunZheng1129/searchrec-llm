"""Pure data-loading helpers for the Streamlit dashboard."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from app.api.service import COMMAND_HINTS
from src.utils.config import load_yaml_config, resolve_project_path


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolve_project_path(str(candidate))


def load_dashboard_config(path: str | None = None) -> dict[str, Any]:
    """Load dashboard config with environment override."""
    raw = path or os.environ.get(
        "TIKSEARCHREC_DASHBOARD_CONFIG",
        "configs/app/dashboard_debug.yaml",
    )
    return load_yaml_config(_resolve(raw))


def load_parquet_if_exists(path: str | Path) -> pd.DataFrame | None:
    """Read parquet if present."""
    resolved = _resolve(path)
    return pd.read_parquet(resolved) if resolved.exists() else None


def load_csv_if_exists(path: str | Path) -> pd.DataFrame | None:
    """Read CSV if present."""
    resolved = _resolve(path)
    return pd.read_csv(resolved) if resolved.exists() else None


def load_markdown_if_exists(path: str | Path) -> str | None:
    """Read markdown text if present."""
    resolved = _resolve(path)
    return resolved.read_text(encoding="utf-8") if resolved.exists() else None


def artifact_status(config: dict[str, Any]) -> pd.DataFrame:
    """Return artifact status as a DataFrame."""
    rows = []
    for key, raw_path in sorted(config.get("artifacts", {}).items()):
        resolved = _resolve(raw_path)
        rows.append(
            {
                "name": key,
                "path": str(raw_path),
                "exists": resolved.exists(),
                "command_hint": COMMAND_HINTS.get(key),
            }
        )
    return pd.DataFrame(rows)


def load_all_demo_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    """Load all configured local artifacts that exist."""
    artifacts: dict[str, Any] = {}
    for key, raw_path in config.get("artifacts", {}).items():
        path = Path(str(raw_path))
        if path.suffix == ".parquet":
            artifacts[key] = load_parquet_if_exists(raw_path)
        elif path.suffix == ".csv":
            artifacts[key] = load_csv_if_exists(raw_path)
        elif path.suffix in {".md", ".txt"}:
            artifacts[key] = load_markdown_if_exists(raw_path)
    return artifacts
