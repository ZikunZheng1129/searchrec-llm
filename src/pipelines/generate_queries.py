"""Generate Stage 2 synthetic query-item pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.query_generation import (  # noqa: E402
    generate_query_item_pairs,
    validate_query_item_pairs,
)
from src.query_understanding.query_parser import parse_query  # noqa: E402
from src.utils.config import load_yaml_config, resolve_project_path  # noqa: E402
from src.utils.io import read_parquet, write_parquet  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolve_project_path(str(candidate))


def _require_stage1_outputs(input_config: dict[str, Any]) -> dict[str, Path]:
    paths = {name: _resolve_path(path) for name, path in input_config.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Stage 1 output files are required before generating queries. Missing: "
            + ", ".join(missing)
        )
    return paths


def _apply_query_understanding(query_item_pairs: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    known_categories = sorted(items["category"].dropna().astype(str).unique().tolist())
    known_brands = sorted(items["brand"].dropna().astype(str).unique().tolist())
    output = query_item_pairs.copy()

    parsed_rows = [
        parse_query(
            query_text=str(query_text),
            known_categories=known_categories,
            known_brands=known_brands,
        )
        for query_text in output["query_text"].tolist()
    ]

    for index, parsed in enumerate(parsed_rows):
        output.at[index, "intent"] = parsed["intent"]
        output.at[index, "category"] = parsed["category"] or output.at[index, "category"]
        output.at[index, "brand"] = parsed["brand"] or output.at[index, "brand"]
        output.at[index, "price_constraint"] = (
            parsed["price_constraint"] or output.at[index, "price_constraint"]
        )
        output.at[index, "use_case"] = parsed["use_case"] or output.at[index, "use_case"]

    return output


def run_generate_queries(config_path: str | Path) -> dict[str, Any]:
    """Run Stage 2 query generation and return a concise summary."""
    logger = get_logger("tiksearchrec.query_generation")
    config = load_yaml_config(config_path)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    input_paths = _require_stage1_outputs(config.get("input", {}))
    items = read_parquet(input_paths["item_metadata_path"])
    query_item_pairs = generate_query_item_pairs(items=items, config=config, seed=seed)
    query_item_pairs = _apply_query_understanding(query_item_pairs, items)
    validate_query_item_pairs(query_item_pairs)

    output_path = _resolve_path(config.get("output", {})["query_item_pairs_path"])
    write_parquet(query_item_pairs, output_path)

    intent_distribution = query_item_pairs["intent"].value_counts().sort_index().to_dict()
    split_distribution = query_item_pairs["split"].value_counts().sort_index().to_dict()
    summary = {
        "num_items": len(items),
        "num_generated_queries": len(query_item_pairs),
        "num_unique_query_texts": int(query_item_pairs["query_text"].nunique()),
        "intent_distribution": intent_distribution,
        "split_distribution": split_distribution,
        "output_path": str(output_path),
    }

    logger.info(
        "Generated queries: items=%s queries=%s unique_texts=%s",
        summary["num_items"],
        summary["num_generated_queries"],
        summary["num_unique_query_texts"],
    )
    logger.info("Intent distribution: %s", intent_distribution)
    logger.info("Split distribution: %s", split_distribution)
    logger.info("Saved query-item pairs to %s", output_path)

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate Stage 2 synthetic query-item pairs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/query_generation_debug.yaml"),
        help="Path to the YAML query generation config.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_generate_queries(args.config)


if __name__ == "__main__":
    main()
