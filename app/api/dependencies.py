"""FastAPI dependency helpers for the local demo service."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from app.api.service import DemoService
from src.utils.config import load_yaml_config, resolve_project_path


@lru_cache(maxsize=1)
def get_config() -> dict:
    """Load the API config, allowing environment override."""
    raw_path = os.environ.get("TIKSEARCHREC_API_CONFIG", "configs/app/api_debug.yaml")
    path = Path(raw_path)
    resolved = path if path.is_absolute() else resolve_project_path(str(path))
    return load_yaml_config(resolved)


@lru_cache(maxsize=1)
def get_demo_service() -> DemoService:
    """Return a cached demo service instance."""
    return DemoService(get_config())
