from pathlib import Path

import pandas as pd
import pytest

from src.utils.io import ensure_dir, read_json, read_parquet, write_json, write_parquet


def test_ensure_dir_creates_directory(tmp_path: Path) -> None:
    directory = ensure_dir(tmp_path / "nested" / "directory")

    assert directory.exists()
    assert directory.is_dir()


def test_json_write_read_roundtrip(tmp_path: Path) -> None:
    obj = {"query": "wireless earbuds", "top_k": 10}
    path = tmp_path / "outputs" / "sample.json"

    write_json(obj, path)

    assert read_json(path) == obj


def test_parquet_write_read_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "item_id": [101, 102],
            "score": [0.25, 0.75],
            "title": ["running shoes", "phone case"],
        }
    )
    path = tmp_path / "tables" / "sample.parquet"

    write_parquet(df, path)
    result = read_parquet(path)

    pd.testing.assert_frame_equal(result, df)


def test_read_missing_json_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_json(tmp_path / "missing.json")
