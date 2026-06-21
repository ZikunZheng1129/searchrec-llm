"""Generate Stage 5 validation artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.report import generate_validation_artifacts  # noqa: E402
from src.utils.config import load_yaml_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def run_generate_validation_report(config_path: str | Path) -> dict[str, str]:
    """Load config and generate Stage 5 artifacts."""
    logger = get_logger("tiksearchrec.generate_validation_report")
    config = load_yaml_config(config_path)
    artifacts = generate_validation_artifacts(config, config_path=config_path)
    for name, path in artifacts.items():
        logger.info("Generated %s: %s", name, path)
    return artifacts


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate Stage 5 validation reports.")
    parser.add_argument("--config", type=Path, required=True, help="Path to Stage 5 config.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_generate_validation_report(args.config)


if __name__ == "__main__":
    main()
