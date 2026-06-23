"""Evaluate Stage 3 retrieval baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_eval import (  # noqa: E402
    evaluate_retriever,
    summarize_retrieval_metrics,
)
from src.pipelines.build_index import (  # noqa: E402
    build_retriever_from_config,
    load_retriever,
    require_input_paths,
    resolve_path,
)
from src.utils.config import load_yaml_config  # noqa: E402
from src.utils.io import read_parquet  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

RESULT_COLUMNS = [
    "method",
    "split",
    "num_queries",
    "top_k",
    "recall_at_5",
    "recall_at_10",
    "recall_at_20",
    "recall_at_50",
    "recall_at_100",
    "mrr_at_5",
    "mrr_at_10",
    "mrr_at_20",
    "mrr_at_50",
    "mrr_at_100",
    "query_coverage",
    "avg_latency_ms",
    "p95_latency_ms",
    "index_backend",
    "config_path",
]


def _load_or_build_retriever(config: dict[str, Any], items: pd.DataFrame) -> Any:
    method = str(config.get("retrieval", {}).get("method"))
    index_path = resolve_path(config.get("output", {})["index_path"])
    if index_path.exists():
        return load_retriever(method, index_path)
    retriever = build_retriever_from_config(config, items)
    retriever.save(index_path)
    return retriever


def _write_summary_row(summary: pd.DataFrame, results_path: Path, config_path: str | Path) -> None:
    output = summary.copy()
    output["config_path"] = str(config_path)
    for column in RESULT_COLUMNS:
        if column not in output.columns:
            text_columns = {"method", "split", "index_backend", "config_path"}
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

    results_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(results_path, index=False)


def run_evaluate_retrieval(config_path: str | Path) -> dict[str, Any]:
    """Run retrieval evaluation and save one summary row."""
    logger = get_logger("tiksearchrec.evaluate_retrieval")
    config = load_yaml_config(config_path)
    set_seed(int(config.get("seed", 42)))
    input_paths = require_input_paths(config.get("input", {}))
    items = read_parquet(input_paths["item_metadata_path"])
    query_item_pairs = read_parquet(input_paths["query_item_pairs_path"])

    evaluation_config = config.get("evaluation", {})
    split = str(evaluation_config.get("split", "test"))
    k_values = [int(k) for k in evaluation_config.get("k_values", [10, 20, 50])]
    top_k = int(config.get("retrieval", {}).get("top_k", max(k_values)))
    filtered_pairs = query_item_pairs[query_item_pairs["split"] == split].reset_index(drop=True)
    if filtered_pairs.empty:
        raise ValueError(f"No query-item pairs found for split={split}")

    retriever = _load_or_build_retriever(config, items)
    per_query = evaluate_retriever(
        retriever=retriever,
        query_item_pairs=filtered_pairs,
        k_values=k_values,
        top_k=top_k,
    )
    summary = summarize_retrieval_metrics(per_query)
    results_path = resolve_path(config.get("output", {})["results_path"])
    _write_summary_row(summary, results_path, config_path)

    summary_row = summary.iloc[0].to_dict()
    summary_row["config_path"] = str(config_path)
    summary_row["results_path"] = str(results_path)
    logger.info(
        "Evaluated %s on split=%s with %s queries",
        summary_row["method"],
        summary_row["split"],
        summary_row["num_queries"],
    )
    logger.info("Results saved to %s", results_path)
    logger.info("Summary: %s", summary_row)
    return summary_row


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a Stage 3 retrieval baseline.")
    parser.add_argument("--config", type=Path, required=True, help="Path to retrieval config.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_evaluate_retrieval(args.config)


if __name__ == "__main__":
    main()
