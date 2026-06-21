"""Evaluation metrics for query-item ranking."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def dcg_at_k(labels: list[int], k: int) -> float:
    """Compute DCG for ranked binary labels."""
    return float(
        sum(float(label) / math.log2(rank + 1) for rank, label in enumerate(labels[:k], start=1))
    )


def ndcg_at_k(labels: list[int], k: int) -> float:
    """Compute NDCG for ranked binary labels."""
    ideal = sorted([int(label) for label in labels], reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    return dcg_at_k(labels, k) / ideal_dcg if ideal_dcg > 0 else 0.0


def mrr_at_k(labels: list[int], k: int) -> float:
    """Compute MRR for ranked binary labels."""
    for rank, label in enumerate(labels[:k], start=1):
        if int(label) > 0:
            return 1.0 / rank
    return 0.0


def precision_at_k(labels: list[int], k: int) -> float:
    """Compute precision at K."""
    if k <= 0:
        return 0.0
    return float(sum(int(label) for label in labels[:k]) / k)


def recall_at_k(labels: list[int], total_relevant: int, k: int) -> float:
    """Compute recall at K."""
    if total_relevant <= 0:
        return 0.0
    return float(sum(int(label) for label in labels[:k]) / total_relevant)


def auc_score(labels: list[int], scores: list[float]) -> float:
    """Compute binary AUC, returning NaN when undefined."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    positives = s[y == 1]
    negatives = s[y == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    wins = 0.0
    for positive in positives:
        wins += float((positive > negatives).sum())
        wins += 0.5 * float((positive == negatives).sum())
    return float(wins / (len(positives) * len(negatives)))


def evaluate_ranking_predictions(
    scored_candidates: pd.DataFrame,
    k_values: list[int],
    score_column: str,
    group_column: str = "query_id",
    label_column: str = "label",
) -> pd.DataFrame:
    """Evaluate scored candidates per query."""
    if scored_candidates.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    split = (
        str(scored_candidates["split"].iloc[0])
        if "split" in scored_candidates.columns and not scored_candidates.empty
        else "unknown"
    )
    method = (
        str(scored_candidates["method"].iloc[0])
        if "method" in scored_candidates.columns and not scored_candidates.empty
        else "unknown"
    )
    for query_id, group in scored_candidates.groupby(group_column, sort=True):
        sort_columns = [score_column]
        ascending = [False]
        if "candidate_item_id" in group.columns:
            sort_columns.append("candidate_item_id")
            ascending.append(True)
        ranked = group.sort_values(sort_columns, ascending=ascending)
        labels = ranked[label_column].astype(int).tolist()
        scores = ranked[score_column].astype(float).tolist()
        total_relevant = int(group[label_column].astype(int).sum())
        row: dict[str, Any] = {
            "query_id": str(query_id),
            "method": method,
            "split": split,
            "num_candidates": int(len(group)),
            "num_relevant": total_relevant,
            "candidate_covered": float(total_relevant > 0),
            "auc": auc_score(labels, scores),
            "latency_ms": float(group["latency_ms"].iloc[0]) if "latency_ms" in group else 0.0,
        }
        for k in k_values:
            row[f"ndcg_at_{k}"] = ndcg_at_k(labels, k)
            row[f"mrr_at_{k}"] = mrr_at_k(labels, k)
            row[f"precision_at_{k}"] = precision_at_k(labels, k)
            row[f"recall_at_{k}"] = recall_at_k(labels, total_relevant, k)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_ranking_metrics(per_query_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-query ranking metrics into one row."""
    if per_query_results.empty:
        return pd.DataFrame(
            [
                {
                    "method": "unknown",
                    "split": "unknown",
                    "num_queries": 0,
                    "top_k": 0,
                    "auc": float("nan"),
                    "candidate_coverage": 0.0,
                    "avg_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                }
            ]
        )
    metric_prefixes = ("ndcg_at_", "mrr_at_", "precision_at_", "recall_at_")
    metric_columns = [
        column for column in per_query_results.columns if column.startswith(metric_prefixes)
    ]
    k_values = [
        int(column.rsplit("_", 1)[1])
        for column in metric_columns
        if column.rsplit("_", 1)[1].isdigit()
    ]
    summary: dict[str, Any] = {
        "method": per_query_results["method"].iloc[0],
        "split": per_query_results["split"].iloc[0],
        "num_queries": int(len(per_query_results)),
        "top_k": int(max(k_values) if k_values else 0),
        "auc": float(per_query_results["auc"].mean(skipna=True)),
        "candidate_coverage": float(per_query_results["candidate_covered"].mean()),
        "avg_latency_ms": float(per_query_results["latency_ms"].mean()),
        "p95_latency_ms": float(np.percentile(per_query_results["latency_ms"], 95)),
    }
    for column in sorted(metric_columns):
        summary[column] = float(per_query_results[column].mean())
    return pd.DataFrame([summary])
