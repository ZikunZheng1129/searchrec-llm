"""Train Stage 6 sequence models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sequence_models.trainer import train_sequence_model  # noqa: E402
from src.utils.config import load_yaml_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def run_train_sequence_model(config_path: str | Path) -> dict:
    """Train a configured sequence model."""
    logger = get_logger("tiksearchrec.train_sequence_model")
    config = load_yaml_config(config_path)
    summary = train_sequence_model(config)
    logger.info(
        "Trained %s on %s with %s items, %s examples, final_loss=%.4f",
        summary["method"],
        summary["device"],
        summary["num_items"],
        summary["num_train_examples"],
        summary["final_train_loss"],
    )
    logger.info("Checkpoint saved to %s", summary["checkpoint_path"])
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a Stage 6 sequence model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to sequence config.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_train_sequence_model(args.config)


if __name__ == "__main__":
    main()
