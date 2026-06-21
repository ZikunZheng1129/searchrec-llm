"""Recommendation baseline evaluation metrics."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd


def hit_rate_at_k(recommended_items: list[str], target_items: set[str], k: int) -> float:
    """Return 1.0 if any target item appears in the top-k recommendations."""
    if not target_items:
        return 0.0
    top_items = {str(item_id) for item_id in recommended_items[:k]}
    return 1.0 if top_items & {str(item_id) for item_id in target_items} else 0.0


def recall_at_k(recommended_items: list[str], target_items: set[str], k: int) -> float:
    """Compute average per-user recall at K."""
    if not target_items:
        return 0.0
    targets = {str(item_id) for item_id in target_items}
    top_items = {str(item_id) for item_id in recommended_items[:k]}
    return len(top_items & targets) / len(targets)


def ndcg_at_k(recommended_items: list[str], target_items: set[str], k: int) -> float:
    """Compute binary-relevance NDCG at K."""
    if not target_items:
        return 0.0
    targets = {str(item_id) for item_id in target_items}
    dcg = 0.0
    for rank, item_id in enumerate(recommended_items[:k], start=1):
        if str(item_id) in targets:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(targets), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(recommended_items: list[str], target_items: set[str], k: int) -> float:
    """Compute reciprocal rank of the first target item in top K."""
    if not target_items:
        return 0.0
    targets = {str(item_id) for item_id in target_items}
    for rank, item_id in enumerate(recommended_items[:k], start=1):
        if str(item_id) in targets:
            return 1.0 / rank
    return 0.0


def coverage_at_k(
    all_recommendations: dict[str, list[str]],
    all_item_ids: set[str],
    k: int,
) -> float:
    """Compute catalog coverage from top-k recommendation lists."""
    if not all_item_ids:
        return 0.0
    recommended = {
        str(item_id)
        for recommendations in all_recommendations.values()
        for item_id in recommendations[:k]
    }
    return len(recommended) / len({str(item_id) for item_id in all_item_ids})


def _method_name(recommender: Any) -> str:
    return str(getattr(recommender, "method_name", recommender.__class__.__name__.lower()))


def evaluate_recommender(
    recommender: Any,
    eval_interactions: pd.DataFrame,
    train_interactions: pd.DataFrame,
    items: pd.DataFrame,
    k_values: list[int],
    top_k: int,
    exclude_seen: bool = True,
) -> pd.DataFrame:
    """Evaluate one recommender and return per-user metric rows."""
    if eval_interactions.empty:
        return pd.DataFrame()

    split = (
        str(eval_interactions["split"].iloc[0])
        if "split" in eval_interactions.columns and not eval_interactions.empty
        else "unknown"
    )
    all_item_ids = set(items["item_id"].dropna().astype(str).tolist())
    rows: list[dict[str, Any]] = []
    all_recommendations: dict[str, list[str]] = {}

    for user_id, group in eval_interactions.groupby("user_id", sort=True):
        user_id = str(user_id)
        target_items = {str(item_id) for item_id in group["item_id"].dropna().astype(str)}
        start_time = time.perf_counter()
        recommendations = recommender.recommend(
            user_id=user_id,
            top_k=top_k,
            exclude_seen=exclude_seen,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        recommended_item_ids = [str(row["item_id"]) for row in recommendations]
        all_recommendations[user_id] = recommended_item_ids

        result_row: dict[str, Any] = {
            "user_id": user_id,
            "method": _method_name(recommender),
            "split": split,
            "target_items": sorted(target_items),
            "recommended_item_ids": recommended_item_ids,
            "num_targets": len(target_items),
            "num_recommendations": len(recommended_item_ids),
            "top_k": int(top_k),
            "latency_ms": latency_ms,
        }
        for k in k_values:
            result_row[f"hit_rate_at_{k}"] = hit_rate_at_k(recommended_item_ids, target_items, k)
            result_row[f"recall_at_{k}"] = recall_at_k(recommended_item_ids, target_items, k)
            result_row[f"ndcg_at_{k}"] = ndcg_at_k(recommended_item_ids, target_items, k)
            result_row[f"mrr_at_{k}"] = mrr_at_k(recommended_item_ids, target_items, k)
        rows.append(result_row)

    per_user = pd.DataFrame(rows)
    for k in k_values:
        per_user[f"coverage_at_{k}"] = coverage_at_k(all_recommendations, all_item_ids, k)
    return per_user


def summarize_recommendation_metrics(per_user_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-user recommendation metrics into one row."""
    if per_user_results.empty:
        return pd.DataFrame(
            [
                {
                    "method": "unknown",
                    "split": "unknown",
                    "num_users": 0,
                    "top_k": 0,
                    "avg_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                }
            ]
        )

    metric_prefixes = (
        "hit_rate_at_",
        "recall_at_",
        "ndcg_at_",
        "mrr_at_",
        "coverage_at_",
    )
    metric_columns = [
        column for column in per_user_results.columns if column.startswith(metric_prefixes)
    ]
    summary = {
        "method": per_user_results["method"].iloc[0],
        "split": per_user_results["split"].iloc[0],
        "num_users": int(len(per_user_results)),
        "top_k": int(per_user_results["top_k"].iloc[0]),
        "avg_latency_ms": float(per_user_results["latency_ms"].mean()),
        "p95_latency_ms": float(np.percentile(per_user_results["latency_ms"], 95)),
    }
    for column in sorted(metric_columns):
        if column.startswith("coverage_at_"):
            summary[column] = float(per_user_results[column].iloc[0])
        else:
            summary[column] = float(per_user_results[column].mean())
    return pd.DataFrame([summary])
