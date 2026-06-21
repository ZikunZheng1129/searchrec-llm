"""Generate Stage 8 behavior-based user profiles."""

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
    evaluate_user_profile_outputs,
    summarize_llm_metrics,
)
from src.llm.user_modeling.profile_pipeline import run_user_profile_generation  # noqa: E402
from src.utils.config import load_yaml_config, resolve_project_path  # noqa: E402
from src.utils.io import read_parquet, write_parquet  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

RESULT_COLUMNS = [
    "stage",
    "method",
    "provider",
    "model",
    "num_users",
    "profile_generation_success_rate",
    "schema_valid_rate",
    "profile_nonempty_rate",
    "embedding_coverage",
    "avg_profile_length",
    "avg_latency_ms",
    "p95_latency_ms",
    "estimated_cost_per_1000_users",
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
            text_columns = {"stage", "method", "provider", "model", "config_path"}
            output[column] = "" if column in text_columns else 0.0
    output = output[RESULT_COLUMNS]
    if results_path.exists():
        existing = pd.read_csv(results_path)
        if set(RESULT_COLUMNS).issubset(existing.columns):
            same_run = (
                (existing["method"] == output["method"].iloc[0])
                & (existing["provider"] == output["provider"].iloc[0])
                & (existing["model"] == output["model"].iloc[0])
                & (existing["config_path"] == output["config_path"].iloc[0])
            )
            existing = existing.loc[~same_run, RESULT_COLUMNS]
        else:
            existing = pd.DataFrame(columns=RESULT_COLUMNS)
        output = pd.concat([existing, output], ignore_index=True)
    output = output.sort_values(["method", "provider", "model", "config_path"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(results_path, index=False)


def run_generate_user_profiles(config_path: str | Path) -> dict[str, Any]:
    """Generate profiles, embeddings, and validation metrics."""
    logger = get_logger("tiksearchrec.generate_user_profiles")
    config = load_yaml_config(config_path)
    set_seed(int(config.get("seed", 42)))
    input_config = config.get("input", {})
    train = read_parquet(_resolve_path(input_config["train_path"]))
    items = read_parquet(_resolve_path(input_config["item_metadata_path"]))
    profiles, embeddings = run_user_profile_generation(train, items, config)

    profiles_path = _resolve_path(config.get("output", {})["user_profiles_path"])
    embeddings_path = _resolve_path(config.get("output", {})["user_profile_embeddings_path"])
    write_parquet(profiles, profiles_path)
    write_parquet(embeddings, embeddings_path)

    per_user = evaluate_user_profile_outputs(profiles, embeddings)
    summary = summarize_llm_metrics(per_user)
    results_path = _resolve_path(config.get("output", {})["results_path"])
    _write_results(summary, results_path, config_path)
    row = summary.iloc[0].to_dict()
    row["config_path"] = str(config_path)
    row["user_profiles_path"] = str(profiles_path)
    row["user_profile_embeddings_path"] = str(embeddings_path)
    row["results_path"] = str(results_path)
    logger.info(
        "Generated user profiles provider=%s model=%s users=%s",
        row["provider"],
        row["model"],
        row["num_users"],
    )
    logger.info(
        "success=%s nonempty=%s embedding=%s avg_len=%s latency_ms=%s",
        row["profile_generation_success_rate"],
        row["profile_nonempty_rate"],
        row["embedding_coverage"],
        row["avg_profile_length"],
        row["avg_latency_ms"],
    )
    logger.info("Profiles saved to %s", profiles_path)
    logger.info("Embeddings saved to %s", embeddings_path)
    logger.info("Results saved to %s", results_path)
    return row


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate Stage 8 user profiles.")
    parser.add_argument("--config", type=Path, required=True, help="Path to user profile config.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_generate_user_profiles(args.config)


if __name__ == "__main__":
    main()
