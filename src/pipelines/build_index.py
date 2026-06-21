"""Build Stage 3 retrieval indexes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from src.retrieval.dense_retriever import TfidfDenseRetriever  # noqa: E402
from src.retrieval.faiss_retriever import FaissRetriever  # noqa: E402
from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from src.utils.config import load_yaml_config, resolve_project_path  # noqa: E402
from src.utils.io import read_parquet  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolve_project_path(str(candidate))


def require_input_paths(input_config: dict[str, Any]) -> dict[str, Path]:
    """Resolve and validate required input files."""
    paths = {name: resolve_path(path) for name, path in input_config.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Required retrieval input files are missing: " + ", ".join(missing))
    return paths


def create_retriever(config: dict[str, Any]) -> Any:
    """Create a retrieval object from config."""
    retrieval_config = config.get("retrieval", {})
    method = retrieval_config.get("method")
    item_text_fields = retrieval_config.get("item_text_fields")

    if method == "bm25":
        bm25_config = config.get("bm25", {})
        return BM25Retriever(
            k1=float(bm25_config.get("k1", 1.5)),
            b=float(bm25_config.get("b", 0.75)),
            item_text_fields=item_text_fields,
        )

    if method == "dense":
        dense_config = config.get("dense", {})
        return TfidfDenseRetriever(
            max_features=int(dense_config.get("max_features", 5000)),
            normalize=bool(dense_config.get("normalize", True)),
            item_text_fields=item_text_fields,
        )

    if method == "faiss":
        dense_config = config.get("dense", {})
        faiss_config = config.get("faiss", {})
        return FaissRetriever(
            backend=str(faiss_config.get("backend", "auto")),
            metric=str(faiss_config.get("metric", "cosine")),
            fallback_to_numpy=bool(faiss_config.get("fallback_to_numpy", True)),
            max_features=int(dense_config.get("max_features", 5000)),
            normalize=bool(dense_config.get("normalize", True)),
            item_text_fields=item_text_fields,
        )

    if method == "hybrid":
        bm25_config = config.get("bm25", {})
        dense_config = config.get("dense", {})
        hybrid_config = config.get("hybrid", {})
        return HybridRetriever(
            bm25_weight=float(hybrid_config.get("bm25_weight", 0.5)),
            dense_weight=float(hybrid_config.get("dense_weight", 0.5)),
            score_normalization=str(hybrid_config.get("score_normalization", "minmax")),
            bm25_params={
                "k1": float(bm25_config.get("k1", 1.5)),
                "b": float(bm25_config.get("b", 0.75)),
                "item_text_fields": item_text_fields,
            },
            dense_params={
                "max_features": int(dense_config.get("max_features", 5000)),
                "normalize": bool(dense_config.get("normalize", True)),
                "item_text_fields": item_text_fields,
            },
        )

    raise ValueError(f"Unsupported retrieval.method: {method}")


def load_retriever(method: str, path: str | Path) -> Any:
    """Load a retriever index for a configured method."""
    if method == "bm25":
        return BM25Retriever.load(path)
    if method == "dense":
        return TfidfDenseRetriever.load(path)
    if method == "faiss":
        return FaissRetriever.load(path)
    if method == "hybrid":
        return HybridRetriever.load(path)
    raise ValueError(f"Unsupported retrieval.method: {method}")


def build_retriever_from_config(config: dict[str, Any], items: pd.DataFrame) -> Any:
    """Create and fit a retriever from config."""
    retriever = create_retriever(config)
    retriever.fit(items)
    return retriever


def run_build_index(config_path: str | Path) -> dict[str, Any]:
    """Run index building and return a concise summary."""
    logger = get_logger("tiksearchrec.build_index")
    config = load_yaml_config(config_path)
    set_seed(int(config.get("seed", 42)))
    input_paths = require_input_paths(config.get("input", {}))
    items = read_parquet(input_paths["item_metadata_path"])
    retriever = build_retriever_from_config(config, items)

    method = str(config.get("retrieval", {}).get("method"))
    index_path = resolve_path(config.get("output", {})["index_path"])
    retriever.save(index_path)
    backend = str(getattr(retriever, "backend", getattr(retriever, "index_backend", "local")))

    summary = {
        "method": method,
        "num_items": len(items),
        "index_path": str(index_path),
        "index_backend": backend,
    }
    logger.info(
        "Built %s index with %s items at %s",
        summary["method"],
        summary["num_items"],
        summary["index_path"],
    )
    logger.info("Index backend: %s", backend)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build a Stage 3 retrieval index.")
    parser.add_argument("--config", type=Path, required=True, help="Path to retrieval config.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_build_index(args.config)


if __name__ == "__main__":
    main()
