"""Evaluate Stage 9 multimodal retrieval-style baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.multimodal_eval import (  # noqa: E402
    evaluate_multimodal_retriever,
    summarize_multimodal_metrics,
)
from src.multimodal.cold_start_model import (  # noqa: E402
    identify_cold_start_items,
    identify_long_tail_items,
)
from src.multimodal.fusion_model import (  # noqa: E402
    MultimodalFusionEncoder,
    build_encoder_from_config,
)
from src.multimodal.multimodal_retriever import MultimodalRetriever  # noqa: E402
from src.utils.config import load_yaml_config, resolve_project_path  # noqa: E402
from src.utils.io import read_parquet, write_parquet  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

RESULT_COLUMNS = [
    "stage",
    "method",
    "split",
    "num_queries",
    "num_items",
    "embedding_dim",
    "recall_at_10",
    "recall_at_20",
    "recall_at_50",
    "ndcg_at_10",
    "ndcg_at_20",
    "ndcg_at_50",
    "mrr_at_10",
    "mrr_at_20",
    "mrr_at_50",
    "cold_start_recall_at_10",
    "cold_start_recall_at_20",
    "long_tail_coverage_at_10",
    "long_tail_coverage_at_20",
    "catalog_coverage_at_10",
    "catalog_coverage_at_20",
    "category_diversity_at_10",
    "category_diversity_at_20",
    "image_available_rate",
    "avg_latency_ms",
    "p95_latency_ms",
    "config_path",
]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolve_project_path(str(candidate))


def _write_results(row: dict[str, Any], results_path: Path) -> None:
    output = pd.DataFrame([row])
    for column in RESULT_COLUMNS:
        if column not in output.columns:
            text_columns = {"stage", "method", "split", "config_path"}
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
    output = output.sort_values(["method", "split", "config_path"]).reset_index(drop=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(results_path, index=False)


def _write_embedding_snapshot(
    items: pd.DataFrame,
    embeddings,
    method: str,
    image_rate: float,
    output_path: Path,
) -> None:
    output = pd.DataFrame(
        {
            "item_id": items["item_id"].astype(str),
            "embedding": [row.astype(float).tolist() for row in embeddings],
            "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            "method": method,
            "has_text_features": method
            in {"text_only", "text_metadata_fusion", "multimodal_fusion"},
            "has_metadata_features": method
            in {"metadata_only", "text_metadata_fusion", "multimodal_fusion"},
            "has_image_features": image_rate > 0.0,
            "image_available": image_rate > 0.0,
        }
    )
    write_parquet(output, output_path)


def run_evaluate_multimodal(config_path: str | Path) -> dict[str, Any]:
    """Evaluate one Stage 9 multimodal config."""
    logger = get_logger("tiksearchrec.evaluate_multimodal")
    config = load_yaml_config(config_path)
    set_seed(int(config.get("seed", 42)))
    input_config = config.get("input", {})
    method = str(config.get("multimodal", {}).get("method"))
    split = str(config.get("multimodal", {}).get("split", "test"))
    top_k = int(config.get("multimodal", {}).get("top_k", 50))

    items = (
        read_parquet(_resolve_path(input_config["item_metadata_path"]))
        .sort_values("item_id")
        .reset_index(drop=True)
    )
    query_pairs = read_parquet(_resolve_path(input_config["query_item_pairs_path"]))
    train = read_parquet(_resolve_path(input_config["train_interactions_path"]))
    eval_pairs = (
        query_pairs[query_pairs["split"].astype(str) == split].copy().reset_index(drop=True)
    )

    encoder = build_encoder_from_config(config)
    embeddings = encoder.fit_transform(items)
    image_rate = float(getattr(encoder, "image_available_rate_", 0.0))
    if isinstance(encoder, MultimodalFusionEncoder):
        image_rate = encoder.image_available_rate_
    embedding_path = _resolve_path(config.get("output", {})["item_embeddings_path"])
    _write_embedding_snapshot(items, embeddings, method, image_rate, embedding_path)

    retriever = MultimodalRetriever().fit(items, embeddings, encoder)
    evaluation_config = config.get("evaluation", {})
    k_values = [int(k) for k in evaluation_config.get("k_values", [10, 20, 50])]
    cold = identify_cold_start_items(
        items,
        train,
        float(evaluation_config.get("cold_start_quantile", 0.25)),
    )
    long_tail = identify_long_tail_items(
        items,
        train,
        float(evaluation_config.get("long_tail_quantile", 0.25)),
    )
    per_query = evaluate_multimodal_retriever(
        retriever=retriever,
        query_item_pairs=eval_pairs,
        items=items,
        train_interactions=train,
        k_values=k_values,
        top_k=top_k,
        cold_start_items=cold,
        long_tail_items=long_tail,
    )
    summary = summarize_multimodal_metrics(per_query)
    row = summary.iloc[0].to_dict()
    row.update(
        {
            "stage": "multimodal",
            "method": method,
            "split": split,
            "num_items": int(len(items)),
            "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            "image_available_rate": image_rate,
            "config_path": str(config_path),
        }
    )
    results_path = _resolve_path(config.get("output", {})["results_path"])
    _write_results(row, results_path)
    logger.info(
        "Evaluated multimodal method=%s split=%s queries=%s", method, split, row["num_queries"]
    )
    logger.info(
        "Recall@10=%s NDCG@10=%s MRR@10=%s ColdStart@10=%s Catalog@10=%s LongTail@10=%s",
        row.get("recall_at_10"),
        row.get("ndcg_at_10"),
        row.get("mrr_at_10"),
        row.get("cold_start_recall_at_10"),
        row.get("catalog_coverage_at_10"),
        row.get("long_tail_coverage_at_10"),
    )
    logger.info("Image available rate: %.4f", image_rate)
    logger.info("Results saved to %s", results_path)
    return row


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Stage 9 multimodal retrieval.")
    parser.add_argument("--config", type=Path, required=True, help="Path to multimodal config.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_evaluate_multimodal(args.config)


if __name__ == "__main__":
    main()
