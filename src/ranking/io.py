"""I/O helpers for ranking artifacts and result CSVs."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

RANKING_RESULT_COLUMNS = [
    "method",
    "backend",
    "split",
    "num_queries",
    "top_k",
    "ndcg_at_10",
    "mrr_at_10",
    "precision_at_10",
    "recall_at_10",
    "ndcg_at_20",
    "mrr_at_20",
    "precision_at_20",
    "recall_at_20",
    "auc",
    "candidate_coverage",
    "avg_latency_ms",
    "p95_latency_ms",
    "config_path",
    "model_path",
]


def save_ranker_artifact(obj: Any, path: str | Path) -> None:
    """Save a non-torch ranker artifact with pickle."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        pickle.dump(obj, file)


def load_ranker_artifact(path: str | Path) -> Any:
    """Load a pickle ranker artifact."""
    with Path(path).open("rb") as file:
        return pickle.load(file)


def update_ranking_results(results_row: dict[str, Any], results_path: str | Path) -> None:
    """Replace or append one ranking result row deterministically."""
    output = pd.DataFrame([results_row])
    for column in RANKING_RESULT_COLUMNS:
        if column not in output.columns:
            text_columns = {"method", "backend", "split", "config_path", "model_path"}
            output[column] = "" if column in text_columns else 0.0
    output = output[RANKING_RESULT_COLUMNS]

    path = Path(results_path)
    if path.exists():
        existing = pd.read_csv(path)
        if set(RANKING_RESULT_COLUMNS).issubset(existing.columns):
            same_run = (
                (existing["method"] == output["method"].iloc[0])
                & (existing["split"] == output["split"].iloc[0])
                & (existing["config_path"] == output["config_path"].iloc[0])
            )
            existing = existing.loc[~same_run, RANKING_RESULT_COLUMNS]
        else:
            existing = pd.DataFrame(columns=RANKING_RESULT_COLUMNS)
        output = pd.concat([existing, output], ignore_index=True)

    output = output.sort_values(["method", "split", "config_path", "model_path"]).reset_index(
        drop=True
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
