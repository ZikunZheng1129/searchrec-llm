"""Evaluate Stage 6 sequence models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sequence_models.evaluate import (  # noqa: E402
    evaluate_sequence_model,
    load_sequence_checkpoint,
    summarize_sequence_metrics,
)
from src.sequence_models.trainer import choose_device, resolve_path  # noqa: E402
from src.utils.config import load_yaml_config  # noqa: E402
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
    "checkpoint_path",
]


def _portable_path(path: Path) -> str:
    """Return a repo-relative path when possible for portable result CSVs."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _write_sequence_results(
    summary: pd.DataFrame,
    results_path: Path,
    config_path: str | Path,
    checkpoint_path: Path,
) -> None:
    output = summary.copy()
    output["config_path"] = str(config_path)
    output["checkpoint_path"] = _portable_path(checkpoint_path)
    for column in RESULT_COLUMNS:
        if column not in output.columns:
            text_columns = {"method", "split", "config_path", "checkpoint_path"}
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

    output = output.sort_values(["method", "config_path", "checkpoint_path"]).reset_index(drop=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(results_path, index=False)


def run_evaluate_sequence_model(config_path: str | Path) -> dict[str, Any]:
    """Evaluate a configured sequence model checkpoint."""
    logger = get_logger("tiksearchrec.evaluate_sequence_model")
    config = load_yaml_config(config_path)
    set_seed(int(config.get("seed", 42)))
    device = choose_device(str(config.get("training", {}).get("device", "cpu")))
    checkpoint_path = resolve_path(config.get("output", {})["checkpoint_path"])
    model, mappings, checkpoint_config = load_sequence_checkpoint(
        checkpoint_path,
        device=str(device),
    )

    input_config = config.get("input", {})
    train = read_parquet(resolve_path(input_config["train_path"]))
    val = read_parquet(resolve_path(input_config["val_path"]))
    test = read_parquet(resolve_path(input_config["test_path"]))
    items = read_parquet(resolve_path(input_config["item_metadata_path"]))
    split = str(config.get("sequence", {}).get("eval_split", "test"))
    eval_interactions = test if split == "test" else val if split == "val" else None
    if eval_interactions is None:
        raise ValueError("Stage 6 evaluation split must be 'val' or 'test'")

    per_user = evaluate_sequence_model(
        model=model,
        train_interactions=train,
        eval_interactions=eval_interactions,
        items=items,
        mappings=mappings,
        config=checkpoint_config,
    )
    summary = summarize_sequence_metrics(per_user)
    results_path = resolve_path(config.get("output", {})["results_path"])
    _write_sequence_results(summary, results_path, config_path, checkpoint_path)

    row = summary.iloc[0].to_dict()
    row["config_path"] = str(config_path)
    row["checkpoint_path"] = _portable_path(checkpoint_path)
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
    parser = argparse.ArgumentParser(description="Evaluate a Stage 6 sequence model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to sequence config.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_evaluate_sequence_model(args.config)


if __name__ == "__main__":
    main()
