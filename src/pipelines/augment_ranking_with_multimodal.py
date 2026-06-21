"""Augment Stage 7 ranking candidates with Stage 9 multimodal scores."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.multimodal import ranking_features  # noqa: E402
from src.multimodal.fusion_model import build_encoder_from_config  # noqa: E402
from src.multimodal.multimodal_retriever import MultimodalRetriever  # noqa: E402
from src.utils.config import load_yaml_config, resolve_project_path  # noqa: E402
from src.utils.io import read_parquet, write_parquet  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolve_project_path(str(candidate))


def run_augment_ranking_with_multimodal(config_path: str | Path) -> dict[str, Any]:
    """Add multimodal score/rank columns to ranking candidates."""
    logger = get_logger("tiksearchrec.augment_ranking_with_multimodal")
    config = load_yaml_config(config_path)
    set_seed(int(config.get("seed", 42)))
    input_config = config.get("input", {})
    output_config = config.get("output", {})
    ranking_path = _resolve_path(input_config["ranking_candidates_path"])
    if not ranking_path.exists():
        raise FileNotFoundError(
            f"Ranking candidates file not found: {ranking_path}. Run Stage 7 first."
        )
    items = (
        read_parquet(_resolve_path(input_config["item_metadata_path"]))
        .sort_values("item_id")
        .reset_index(drop=True)
    )
    query_pairs = read_parquet(_resolve_path(input_config["query_item_pairs_path"]))
    ranking_candidates = read_parquet(ranking_path)
    encoder = build_encoder_from_config(config)
    embeddings = encoder.fit_transform(items)
    retriever = MultimodalRetriever().fit(items, embeddings, encoder)
    augmentation = config.get("ranking_augmentation", {})
    output = ranking_features.add_multimodal_scores_to_ranking_candidates(
        ranking_candidates=ranking_candidates,
        query_item_pairs=query_pairs,
        items=items,
        retriever=retriever,
        score_column=str(augmentation.get("score_column", "multimodal_score")),
        rank_column=str(augmentation.get("rank_column", "multimodal_rank")),
    )
    output_path = _resolve_path(output_config["ranking_candidates_output_path"])
    write_parquet(output, output_path)
    scored_rows = int(
        output[str(augmentation.get("score_column", "multimodal_score"))].notna().sum()
    )
    logger.info("Input rows: %s", len(ranking_candidates))
    logger.info("Output rows: %s", len(output))
    logger.info("Scored rows: %s", scored_rows)
    logger.info("Augmented ranking candidates saved to %s", output_path)
    return {
        "input_rows": int(len(ranking_candidates)),
        "output_rows": int(len(output)),
        "scored_rows": scored_rows,
        "output_path": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Augment ranking candidates with multimodal scores."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to multimodal config.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_augment_ranking_with_multimodal(args.config)


if __name__ == "__main__":
    main()
