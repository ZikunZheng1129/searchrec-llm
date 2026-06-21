"""Train Stage 7 ranking models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ranking.cross_encoder_ranker import CrossEncoderRankerWrapper  # noqa: E402
from src.ranking.dataset import get_numeric_feature_columns, split_ranking_candidates  # noqa: E402
from src.ranking.lightgbm_ranker import LightGBMRanker  # noqa: E402
from src.ranking.mixed_ranker import MixedRanker  # noqa: E402
from src.ranking.mlp_ranker import MLPRankerWrapper  # noqa: E402
from src.utils.config import load_yaml_config, resolve_project_path  # noqa: E402
from src.utils.io import read_parquet  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolve_project_path(str(candidate))


def _load_items_if_needed(config: dict[str, Any]) -> Any:
    item_path = config.get("input", {}).get("item_metadata_path")
    return read_parquet(_resolve_path(item_path)) if item_path else None


def create_ranker(config: dict[str, Any]) -> Any:
    """Create a ranker from config."""
    method = str(config.get("ranking", {}).get("method"))
    if method == "lightgbm":
        training = config.get("training", {})
        ranking = config.get("ranking", {})
        return LightGBMRanker(
            backend=str(ranking.get("backend", "auto")),
            fallback_backend=str(ranking.get("fallback_backend", "numpy_linear")),
            learning_rate=float(training.get("learning_rate", 0.05)),
            num_boost_round=int(training.get("num_boost_round", 30)),
            num_epochs_fallback=int(training.get("num_epochs_fallback", 100)),
            regularization=float(training.get("regularization", 0.001)),
            seed=int(config.get("seed", 42)),
        )
    if method == "mlp":
        return MLPRankerWrapper(config)
    if method == "cross_encoder":
        return CrossEncoderRankerWrapper(config)
    if method == "mixed":
        return MixedRanker(**config.get("mixed", {}))
    raise ValueError(f"Unsupported ranking.method: {method}")


def run_train_ranker(config_path: str | Path) -> dict[str, Any]:
    """Train and save a configured ranker."""
    logger = get_logger("tiksearchrec.train_ranker")
    config = load_yaml_config(config_path)
    set_seed(int(config.get("seed", 42)))
    ranking_config = config.get("ranking", {})
    method = str(ranking_config.get("method"))
    label_column = str(ranking_config.get("label_column", "label"))
    group_column = str(ranking_config.get("group_column", "query_id"))

    candidates_path = _resolve_path(config.get("input", {})["ranking_candidates_path"])
    if not candidates_path.exists():
        raise FileNotFoundError(f"Ranking candidates file not found: {candidates_path}")
    candidates = read_parquet(candidates_path)
    train_df = split_ranking_candidates(candidates, "train")
    feature_columns = get_numeric_feature_columns(train_df)
    ranker = create_ranker(config)

    if method == "cross_encoder":
        items = _load_items_if_needed(config)
        if items is None:
            raise ValueError("cross_encoder config must include input.item_metadata_path")
        ranker.fit(train_df, items, label_column=label_column)
    elif method == "lightgbm":
        ranker.fit(
            train_df,
            feature_columns,
            group_column=group_column,
            label_column=label_column,
        )
    elif method == "mlp":
        ranker.fit(train_df, feature_columns, label_column=label_column)
    elif method == "mixed":
        ranker.fit(train_df)

    model_path = _resolve_path(config.get("output", {})["model_path"])
    ranker.save(model_path)
    backend = str(getattr(ranker, "backend", "n/a"))

    logger.info("Trained ranker method=%s backend=%s", method, backend)
    logger.info("Train rows: %s", len(train_df))
    logger.info("Train queries: %s", train_df[group_column].nunique())
    logger.info("Feature count: %s", len(feature_columns) if method != "cross_encoder" else "text")
    logger.info("Model saved to %s", model_path)
    return {
        "method": method,
        "backend": backend,
        "train_rows": int(len(train_df)),
        "train_queries": int(train_df[group_column].nunique()),
        "feature_count": len(feature_columns) if method != "cross_encoder" else "text",
        "model_path": str(model_path),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a Stage 7 ranker.")
    parser.add_argument("--config", type=Path, required=True, help="Path to ranker config.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_train_ranker(args.config)


if __name__ == "__main__":
    main()
