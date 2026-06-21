"""Build Stage 9 local multimodal item embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.multimodal.fusion_model import (  # noqa: E402
    MultimodalFusionEncoder,
    build_encoder_from_config,
)
from src.utils.config import load_yaml_config, resolve_project_path  # noqa: E402
from src.utils.io import read_parquet, write_parquet  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolve_project_path(str(candidate))


def _embedding_frame(
    items: pd.DataFrame,
    embeddings,
    method: str,
    encoder: Any,
) -> pd.DataFrame:
    image_rate = float(getattr(encoder, "image_available_rate_", 0.0))
    has_text = method in {"text_only", "text_metadata_fusion", "multimodal_fusion"}
    has_metadata = method in {"metadata_only", "text_metadata_fusion", "multimodal_fusion"}
    has_image = image_rate > 0.0
    return pd.DataFrame(
        {
            "item_id": items["item_id"].astype(str),
            "embedding": [row.astype(float).tolist() for row in embeddings],
            "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            "method": method,
            "has_text_features": has_text,
            "has_metadata_features": has_metadata,
            "has_image_features": has_image,
            "image_available": has_image,
        }
    )


def run_build_multimodal_embeddings(config_path: str | Path) -> dict[str, Any]:
    """Build and save configured item embeddings."""
    logger = get_logger("tiksearchrec.build_multimodal_embeddings")
    config = load_yaml_config(config_path)
    set_seed(int(config.get("seed", 42)))
    method = str(config.get("multimodal", {}).get("method"))
    items_path = _resolve_path(config.get("input", {})["item_metadata_path"])
    if not items_path.exists():
        raise FileNotFoundError(f"Item metadata file not found: {items_path}")
    items = read_parquet(items_path).sort_values("item_id").reset_index(drop=True)
    encoder = build_encoder_from_config(config)
    embeddings = encoder.fit_transform(items)
    image_rate = float(getattr(encoder, "image_available_rate_", 0.0))
    if isinstance(encoder, MultimodalFusionEncoder):
        image_rate = encoder.image_available_rate_
    output = _embedding_frame(items, embeddings, method, encoder)
    output_path = _resolve_path(config.get("output", {})["item_embeddings_path"])
    write_parquet(output, output_path)
    logger.info("Built %s embeddings for %s items", method, len(items))
    logger.info("Embedding dim: %s", int(embeddings.shape[1]) if embeddings.ndim == 2 else 0)
    logger.info("Image available rate: %.4f", image_rate)
    logger.info("Embeddings saved to %s", output_path)
    return {
        "method": method,
        "num_items": int(len(items)),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "image_available_rate": image_rate,
        "output_path": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build Stage 9 multimodal embeddings.")
    parser.add_argument("--config", type=Path, required=True, help="Path to multimodal config.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_build_multimodal_embeddings(args.config)


if __name__ == "__main__":
    main()
