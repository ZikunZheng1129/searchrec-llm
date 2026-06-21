from pathlib import Path

import pytest

from src.utils.config import load_yaml_config, resolve_project_path, save_yaml_config


def test_yaml_save_load_roundtrip(tmp_path: Path) -> None:
    config = {
        "project": "TikSearchRec-LLM",
        "retrieval": {"top_k": 50},
        "features": ["query_text", "video_caption"],
    }
    path = tmp_path / "configs" / "sample.yaml"

    save_yaml_config(config, path)

    assert load_yaml_config(path) == config


def test_empty_yaml_returns_empty_dict(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    assert load_yaml_config(path) == {}


def test_missing_yaml_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml_config(tmp_path / "missing.yaml")


def test_resolve_project_path_returns_path() -> None:
    path = resolve_project_path("configs")

    assert isinstance(path, Path)
    assert path.is_absolute()
    assert path.name == "configs"
