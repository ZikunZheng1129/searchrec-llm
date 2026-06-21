"""Evaluate Stage 8 LLM query understanding."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.llm_eval import (  # noqa: E402
    evaluate_query_understanding_outputs,
    summarize_llm_metrics,
)
from src.llm.query_understanding.query_understanding_pipeline import (  # noqa: E402
    run_query_understanding,
)
from src.utils.config import load_yaml_config, resolve_project_path  # noqa: E402
from src.utils.io import read_parquet, write_parquet  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

RESULT_COLUMNS = [
    "stage",
    "method",
    "provider",
    "model",
    "split",
    "num_queries",
    "output_parse_success_rate",
    "schema_valid_rate",
    "intent_match_rate",
    "category_match_rate",
    "brand_match_rate",
    "price_constraint_match_rate",
    "use_case_match_rate",
    "avg_expanded_queries",
    "avg_latency_ms",
    "p95_latency_ms",
    "estimated_cost_per_1000_queries",
    "config_path",
]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolve_project_path(str(candidate))


def _write_results(summary: pd.DataFrame, results_path: Path, config_path: str | Path) -> None:
    output = summary.copy()
    output["config_path"] = str(config_path)
    for column in RESULT_COLUMNS:
        if column not in output.columns:
            text_columns = {"stage", "method", "provider", "model", "split", "config_path"}
            output[column] = "" if column in text_columns else 0.0
    output = output[RESULT_COLUMNS]
    if results_path.exists():
        existing = pd.read_csv(results_path)
        if set(RESULT_COLUMNS).issubset(existing.columns):
            same_run = (
                (existing["method"] == output["method"].iloc[0])
                & (existing["provider"] == output["provider"].iloc[0])
                & (existing["model"] == output["model"].iloc[0])
                & (existing["split"] == output["split"].iloc[0])
                & (existing["config_path"] == output["config_path"].iloc[0])
            )
            existing = existing.loc[~same_run, RESULT_COLUMNS]
        else:
            existing = pd.DataFrame(columns=RESULT_COLUMNS)
        output = pd.concat([existing, output], ignore_index=True)
    output = output.sort_values(["method", "provider", "model", "split", "config_path"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(results_path, index=False)


def run_evaluate_llm_query_understanding(config_path: str | Path) -> dict[str, Any]:
    """Run query understanding, write parquet outputs, and update result CSV."""
    logger = get_logger("tiksearchrec.evaluate_llm_query_understanding")
    config = load_yaml_config(config_path)
    set_seed(int(config.get("seed", 42)))
    input_config = config.get("input", {})
    query_pairs = read_parquet(_resolve_path(input_config["query_item_pairs_path"]))
    items = read_parquet(_resolve_path(input_config["item_metadata_path"]))
    parsed = run_query_understanding(query_pairs, items, config)
    parsed_path = _resolve_path(config.get("output", {})["parsed_queries_path"])
    write_parquet(parsed, parsed_path)

    per_query = evaluate_query_understanding_outputs(parsed)
    summary = summarize_llm_metrics(per_query)
    results_path = _resolve_path(config.get("output", {})["results_path"])
    _write_results(summary, results_path, config_path)
    row = summary.iloc[0].to_dict()
    row["config_path"] = str(config_path)
    row["parsed_queries_path"] = str(parsed_path)
    row["results_path"] = str(results_path)
    logger.info(
        "Evaluated LLM query understanding provider=%s model=%s split=%s queries=%s",
        row["provider"],
        row["model"],
        row["split"],
        row["num_queries"],
    )
    logger.info(
        "parse=%s schema=%s intent=%s category=%s latency_ms=%s",
        row["output_parse_success_rate"],
        row["schema_valid_rate"],
        row["intent_match_rate"],
        row["category_match_rate"],
        row["avg_latency_ms"],
    )
    logger.info("Parsed queries saved to %s", parsed_path)
    logger.info("Results saved to %s", results_path)
    return row


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Stage 8 LLM query understanding.")
    parser.add_argument("--config", type=Path, required=True, help="Path to LLM config.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_evaluate_llm_query_understanding(args.config)


if __name__ == "__main__":
    main()
