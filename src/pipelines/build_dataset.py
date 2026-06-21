"""Build the Stage 1 synthetic debug dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (  # noqa: E402
    attach_split_column,
    validate_interactions_schema,
    validate_items_schema,
    validate_users_schema,
)
from src.data.negative_sampling import sample_negative_items  # noqa: E402
from src.data.preprocess import build_debug_dataset  # noqa: E402
from src.data.split import build_user_sequences, leave_one_out_split  # noqa: E402
from src.utils.config import load_yaml_config, resolve_project_path  # noqa: E402
from src.utils.io import write_parquet  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def _resolve_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    return path if path.is_absolute() else resolve_project_path(output_dir)


def run_build_dataset(config_path: str | Path) -> dict[str, Any]:
    """Run the Stage 1 dataset build and return a concise summary."""
    logger = get_logger("tiksearchrec.data_pipeline")
    config = load_yaml_config(config_path)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    users, items, interactions = build_debug_dataset(config)
    validate_users_schema(users)
    validate_items_schema(items)
    validate_interactions_schema(interactions)

    split_config = config.get("split", {})
    if split_config.get("method", "leave_one_out") != "leave_one_out":
        raise ValueError("Stage 1 currently supports only split.method='leave_one_out'")

    train, val, test = leave_one_out_split(
        interactions,
        val_last_n=int(split_config.get("val_last_n", 1)),
        test_last_n=int(split_config.get("test_last_n", 1)),
    )
    combined = attach_split_column(train, val, test)
    train = combined[combined["split"] == "train"].reset_index(drop=True)
    val = combined[combined["split"] == "val"].reset_index(drop=True)
    test = combined[combined["split"] == "test"].reset_index(drop=True)

    user_sequences = build_user_sequences(interactions)
    negative_config = config.get("negative_sampling", {})
    negative_samples = sample_negative_items(
        interactions=combined,
        all_item_ids=items["item_id"].tolist(),
        num_negatives_per_positive=int(negative_config.get("num_negatives_per_positive", 5)),
        seed=seed,
    )

    output_dir = _resolve_output_dir(
        str(config.get("data", {}).get("output_dir", "data/processed"))
    )
    output_paths = {
        "train": output_dir / "train.parquet",
        "val": output_dir / "val.parquet",
        "test": output_dir / "test.parquet",
        "item_metadata": output_dir / "item_metadata.parquet",
        "user_sequences": output_dir / "user_sequences.parquet",
        "negative_samples": output_dir / "negative_samples.parquet",
    }

    write_parquet(train, output_paths["train"])
    write_parquet(val, output_paths["val"])
    write_parquet(test, output_paths["test"])
    write_parquet(items, output_paths["item_metadata"])
    write_parquet(user_sequences, output_paths["user_sequences"])
    write_parquet(negative_samples, output_paths["negative_samples"])

    summary = {
        "num_users": len(users),
        "num_items": len(items),
        "num_interactions": len(interactions),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "num_negative_samples": len(negative_samples),
        "output_paths": {name: str(path) for name, path in output_paths.items()},
    }

    logger.info(
        "Built dataset: users=%s items=%s interactions=%s train=%s val=%s test=%s negatives=%s",
        summary["num_users"],
        summary["num_items"],
        summary["num_interactions"],
        summary["train_size"],
        summary["val_size"],
        summary["test_size"],
        summary["num_negative_samples"],
    )
    for name, path in output_paths.items():
        logger.info("Saved %s to %s", name, path)

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build the Stage 1 debug dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/debug_sample.yaml"),
        help="Path to the YAML data config.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_build_dataset(args.config)


if __name__ == "__main__":
    main()
