"""Evaluate Stage 4 recommendation baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.recommendation_eval import (  # noqa: E402
    evaluate_recommender,
    summarize_recommendation_metrics,
)
from src.recommendation.itemcf import ItemCFRecommender  # noqa: E402
from src.recommendation.matrix_factorization import MatrixFactorizationRecommender  # noqa: E402
from src.recommendation.popularity import PopularityRecommender  # noqa: E402
from src.recommendation.user_history_embedding import UserHistoryEmbeddingRecommender  # noqa: E402
from src.utils.config import load_yaml_config, resolve_project_path  # noqa: E402
from src.utils.io import read_parquet  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

RESULT_COLUMNS = [
    "method",
    "split",
    "num_users",
    "top_k",
    "hit_rate_at_10",
    "recall_at_10",
    "ndcg_at_10",
    "mrr_at_10",
    "hit_rate_at_20",
    "recall_at_20",
    "ndcg_at_20",
    "mrr_at_20",
    "coverage_at_10",
    "coverage_at_20",
    "avg_latency_ms",
    "p95_latency_ms",
    "config_path",
]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolve_project_path(str(candidate))


def _require_inputs(input_config: dict[str, Any]) -> dict[str, Path]:
    paths = {name: _resolve_path(path) for name, path in input_config.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required recommender input files are missing: " + ", ".join(missing)
        )
    return paths


def create_recommender(config: dict[str, Any]) -> Any:
    """Create a Stage 4 recommender from config."""
    method = config.get("recommendation", {}).get("method")
    if method == "popularity":
        popularity_config = config.get("popularity", {})
        return PopularityRecommender(
            score_column=str(popularity_config.get("score_column", "event_weight"))
        )

    if method == "itemcf":
        itemcf_config = config.get("itemcf", {})
        return ItemCFRecommender(
            similarity=str(itemcf_config.get("similarity", "cosine")),
            min_cooccurrence=int(itemcf_config.get("min_cooccurrence", 1)),
            max_neighbors=int(itemcf_config.get("max_neighbors", 50)),
        )

    if method == "matrix_factorization":
        mf_config = config.get("matrix_factorization", {})
        return MatrixFactorizationRecommender(
            factors=int(mf_config.get("factors", 16)),
            epochs=int(mf_config.get("epochs", 20)),
            learning_rate=float(mf_config.get("learning_rate", 0.05)),
            regularization=float(mf_config.get("regularization", 0.01)),
            negative_samples=int(mf_config.get("negative_samples", 3)),
            seed=int(config.get("seed", 42)),
        )

    if method == "user_history_embedding":
        embedding_config = config.get("user_history_embedding", {})
        return UserHistoryEmbeddingRecommender(
            text_fields=list(embedding_config.get("text_fields", [])) or None,
            max_features=int(embedding_config.get("max_features", 5000)),
            normalize=bool(embedding_config.get("normalize", True)),
        )

    raise ValueError(f"Unsupported recommendation.method: {method}")


def _write_summary(summary: pd.DataFrame, results_path: Path, config_path: str | Path) -> None:
    output = summary.copy()
    output["config_path"] = str(config_path)
    for column in RESULT_COLUMNS:
        if column not in output.columns:
            text_columns = {"method", "split", "config_path"}
            output[column] = "" if column in text_columns else 0.0
    output = output[RESULT_COLUMNS]

    if results_path.exists():
        existing = pd.read_csv(results_path)
        if set(RESULT_COLUMNS).issubset(existing.columns):
            same_run = (
                (existing["method"] == output["method"].iloc[0])
                & (existing["split"] == output["split"].iloc[0])
                & (existing["config_path"] == output["config_path"].iloc[0])
            )
            existing = existing.loc[~same_run, RESULT_COLUMNS]
        else:
            existing = pd.DataFrame(columns=RESULT_COLUMNS)
        output = pd.concat([existing, output], ignore_index=True)

    output = output.sort_values(["method", "config_path"]).reset_index(drop=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(results_path, index=False)


def run_evaluate_recommender(config_path: str | Path) -> dict[str, Any]:
    """Fit and evaluate one recommender baseline."""
    logger = get_logger("tiksearchrec.evaluate_recommenders")
    config = load_yaml_config(config_path)
    set_seed(int(config.get("seed", 42)))

    input_paths = _require_inputs(config.get("input", {}))
    train = read_parquet(input_paths["train_path"])
    val = read_parquet(input_paths["val_path"])
    test = read_parquet(input_paths["test_path"])
    items = read_parquet(input_paths["item_metadata_path"])

    split = str(config.get("evaluation", {}).get("split", "test"))
    eval_interactions = test if split == "test" else val if split == "val" else None
    if eval_interactions is None:
        raise ValueError("Stage 4 evaluation split must be 'val' or 'test'")

    recommender = create_recommender(config)
    recommender.fit(train, items)

    k_values = [int(k) for k in config.get("evaluation", {}).get("k_values", [10, 20])]
    top_k = max(int(config.get("recommendation", {}).get("top_k", max(k_values))), max(k_values))
    exclude_seen = bool(config.get("recommendation", {}).get("exclude_seen", True))
    per_user = evaluate_recommender(
        recommender=recommender,
        eval_interactions=eval_interactions,
        train_interactions=train,
        items=items,
        k_values=k_values,
        top_k=top_k,
        exclude_seen=exclude_seen,
    )
    summary = summarize_recommendation_metrics(per_user)
    results_path = _resolve_path(config.get("output", {})["results_path"])
    _write_summary(summary, results_path, config_path)

    row = summary.iloc[0].to_dict()
    row["config_path"] = str(config_path)
    row["results_path"] = str(results_path)
    logger.info(
        "Evaluated %s on split=%s with %s users",
        row["method"],
        row["split"],
        row["num_users"],
    )
    logger.info(
        "HR@10=%s Recall@10=%s NDCG@10=%s MRR@10=%s Coverage@10=%s avg_latency_ms=%s",
        row.get("hit_rate_at_10"),
        row.get("recall_at_10"),
        row.get("ndcg_at_10"),
        row.get("mrr_at_10"),
        row.get("coverage_at_10"),
        row.get("avg_latency_ms"),
    )
    logger.info("Results saved to %s", results_path)
    return row


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a Stage 4 recommender baseline.")
    parser.add_argument("--config", type=Path, required=True, help="Path to recommender config.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_evaluate_recommender(args.config)


if __name__ == "__main__":
    main()
