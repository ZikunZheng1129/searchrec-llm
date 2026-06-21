"""Build Stage 7 ranking candidates and features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ranking.features import build_ranking_features  # noqa: E402
from src.utils.config import load_yaml_config, resolve_project_path  # noqa: E402
from src.utils.io import read_parquet, write_parquet  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolve_project_path(str(candidate))


def _require_inputs(input_config: dict[str, Any]) -> dict[str, Path]:
    paths = {name: _resolve_path(path) for name, path in input_config.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Required ranking input files are missing: " + ", ".join(missing))
    return paths


def run_build_ranking_dataset(config_path: str | Path) -> dict[str, Any]:
    """Build and save the ranking candidate dataset."""
    logger = get_logger("tiksearchrec.build_ranking_dataset")
    config = load_yaml_config(config_path)
    set_seed(int(config.get("seed", 42)))

    input_paths = _require_inputs(config.get("input", {}))
    items = read_parquet(input_paths["item_metadata_path"])
    query_item_pairs = read_parquet(input_paths["query_item_pairs_path"])
    train = read_parquet(input_paths["train_interactions_path"])

    candidate_config = config.get("candidate_generation", {})
    candidates = build_ranking_features(
        query_item_pairs=query_item_pairs,
        items=items,
        train_interactions=train,
        candidate_pool_size=int(candidate_config.get("candidate_pool_size", 50)),
        random_negatives_per_query=int(candidate_config.get("random_negatives_per_query", 10)),
        seed=int(config.get("seed", 42)),
        config=config,
    )

    output_path = _resolve_path(config.get("output", {})["ranking_candidates_path"])
    write_parquet(candidates, output_path)

    num_queries = int(candidates["query_id"].nunique()) if not candidates.empty else 0
    coverage = float(candidates.groupby("query_id")["label"].max().mean()) if num_queries else 0.0
    avg_candidates = float(len(candidates) / num_queries) if num_queries else 0.0
    split_distribution = candidates["split"].value_counts().sort_index().to_dict()

    logger.info("Built ranking candidates for %s queries", num_queries)
    logger.info("Candidate rows: %s", len(candidates))
    logger.info("Split distribution: %s", split_distribution)
    logger.info("Positive candidate coverage: %.4f", coverage)
    logger.info("Average candidates per query: %.2f", avg_candidates)
    logger.info("Ranking candidates saved to %s", output_path)
    return {
        "num_queries": num_queries,
        "num_candidate_rows": int(len(candidates)),
        "split_distribution": split_distribution,
        "positive_candidate_coverage": coverage,
        "avg_candidates_per_query": avg_candidates,
        "output_path": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build Stage 7 ranking candidates.")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to ranking dataset config."
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_build_ranking_dataset(args.config)


if __name__ == "__main__":
    main()
