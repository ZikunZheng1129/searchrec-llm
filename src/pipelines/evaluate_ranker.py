"""Evaluate Stage 7 ranking models."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.ranking_eval import (  # noqa: E402
    auc_score,
    evaluate_ranking_predictions,
    summarize_ranking_metrics,
)
from src.ranking.cross_encoder_ranker import CrossEncoderRankerWrapper  # noqa: E402
from src.ranking.dataset import split_ranking_candidates  # noqa: E402
from src.ranking.io import update_ranking_results  # noqa: E402
from src.ranking.lightgbm_ranker import LightGBMRanker  # noqa: E402
from src.ranking.mixed_ranker import MixedRanker  # noqa: E402
from src.ranking.mlp_ranker import MLPRankerWrapper  # noqa: E402
from src.utils.config import load_yaml_config, resolve_project_path  # noqa: E402
from src.utils.io import read_parquet  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolve_project_path(str(candidate))


def _portable_path(path: Path) -> str:
    """Return a repo-relative path when possible for portable result CSVs."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_ranker(method: str, model_path: Path, device: str = "cpu") -> Any:
    if method == "lightgbm":
        return LightGBMRanker.load(model_path)
    if method == "mlp":
        return MLPRankerWrapper.load(model_path, device=device)
    if method == "cross_encoder":
        return CrossEncoderRankerWrapper.load(model_path, device=device)
    if method == "mixed":
        return MixedRanker.load(model_path)
    raise ValueError(f"Unsupported ranking.method: {method}")


def _score_candidates(
    ranker: Any,
    method: str,
    eval_df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, float]:
    start = time.perf_counter()
    if method == "cross_encoder":
        item_path = config.get("input", {}).get("item_metadata_path")
        if not item_path:
            raise ValueError("cross_encoder config must include input.item_metadata_path")
        items = read_parquet(_resolve_path(item_path))
        scores = ranker.predict(eval_df, items)
    else:
        scores = ranker.predict(eval_df)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    scored = eval_df.copy()
    scored["model_score"] = scores
    scored["method"] = method
    num_queries = max(1, scored["query_id"].nunique())
    scored["latency_ms"] = elapsed_ms / num_queries
    return scored, elapsed_ms


def run_evaluate_ranker(config_path: str | Path) -> dict[str, Any]:
    """Load, score, evaluate, and persist one ranking result row."""
    logger = get_logger("tiksearchrec.evaluate_ranker")
    config = load_yaml_config(config_path)
    set_seed(int(config.get("seed", 42)))
    ranking_config = config.get("ranking", {})
    method = str(ranking_config.get("method"))
    split = str(config.get("evaluation", {}).get("split", "test"))
    device = str(config.get("training", {}).get("device", "cpu"))

    candidates_path = _resolve_path(config.get("input", {})["ranking_candidates_path"])
    model_path = _resolve_path(config.get("output", {})["model_path"])
    if not candidates_path.exists():
        raise FileNotFoundError(f"Ranking candidates file not found: {candidates_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Ranker model artifact not found: {model_path}")

    candidates = read_parquet(candidates_path)
    eval_df = split_ranking_candidates(candidates, split)
    ranker = _load_ranker(method, model_path, device=device)
    scored, _ = _score_candidates(ranker, method, eval_df, config)

    k_values = [int(k) for k in config.get("evaluation", {}).get("k_values", [10, 20])]
    per_query = evaluate_ranking_predictions(
        scored_candidates=scored,
        k_values=k_values,
        score_column="model_score",
        group_column=str(ranking_config.get("group_column", "query_id")),
        label_column=str(ranking_config.get("label_column", "label")),
    )
    summary = summarize_ranking_metrics(per_query)
    row = summary.iloc[0].to_dict()
    row["auc"] = auc_score(
        scored[str(ranking_config.get("label_column", "label"))].astype(int).tolist(),
        scored["model_score"].astype(float).tolist(),
    )
    row["backend"] = str(getattr(ranker, "backend", "n/a"))
    row["config_path"] = str(config_path)
    row["model_path"] = _portable_path(model_path)
    row["top_k"] = int(ranking_config.get("top_k", max(k_values)))

    results_path = _resolve_path(config.get("output", {})["results_path"])
    update_ranking_results(row, results_path)

    logger.info(
        "Evaluated ranker method=%s backend=%s split=%s queries=%s",
        method,
        row["backend"],
        split,
        row["num_queries"],
    )
    logger.info(
        "NDCG@10=%s MRR@10=%s Precision@10=%s Recall@10=%s AUC=%s Coverage=%s latency_ms=%s",
        row.get("ndcg_at_10"),
        row.get("mrr_at_10"),
        row.get("precision_at_10"),
        row.get("recall_at_10"),
        row.get("auc"),
        row.get("candidate_coverage"),
        row.get("avg_latency_ms"),
    )
    logger.info("Results saved to %s", results_path)
    return row


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a Stage 7 ranker.")
    parser.add_argument("--config", type=Path, required=True, help="Path to ranker config.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_evaluate_ranker(args.config)


if __name__ == "__main__":
    main()
