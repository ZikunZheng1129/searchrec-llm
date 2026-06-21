"""Common I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a UTF-8 JSON file containing an object."""
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object at the top level: {json_path}")
    return data


def write_json(obj: dict[str, Any], path: str | Path) -> None:
    """Write a dictionary to a UTF-8 JSON file."""
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(obj, file, indent=2)
        file.write("\n")


def read_parquet(path: str | Path) -> pd.DataFrame:
    """Read a parquet file into a DataFrame."""
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    return pd.read_parquet(parquet_path)


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to parquet."""
    parquet_path = Path(path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
