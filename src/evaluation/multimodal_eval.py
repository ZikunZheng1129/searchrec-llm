"""Evaluation helpers for Stage 9 multimodal item retrieval."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd


def recall_at_k(results: list[str], target_item_id: str, k: int) -> float:
    """Return 1.0 when target is present in top-k results."""
    return 1.0 if str(target_item_id) in {str(item_id) for item_id in results[:k]} else 0.0


def ndcg_at_k(results: list[str], target_item_id: str, k: int) -> float:
    """Compute one-positive NDCG@K."""
    target = str(target_item_id)
    for rank, item_id in enumerate(results[:k], start=1):
        if str(item_id) == target:
            return float(1.0 / math.log2(rank + 1))
    return 0.0


def mrr_at_k(results: list[str], target_item_id: str, k: int) -> float:
    """Compute one-positive MRR@K."""
    target = str(target_item_id)
    for rank, item_id in enumerate(results[:k], start=1):
        if str(item_id) == target:
            return float(1.0 / rank)
    return 0.0


def catalog_coverage_at_k(
    all_results: list[list[str]],
    all_item_ids: set[str],
    k: int,
) -> float:
    """Compute catalog coverage from retrieved top-k items."""
    if not all_item_ids:
        return 0.0
    retrieved = {str(item_id) for results in all_results for item_id in results[:k]}
    return float(len(retrieved) / len({str(item_id) for item_id in all_item_ids}))


def category_diversity_at_k(
    all_results: list[list[str]],
    item_to_category: dict[str, str],
    k: int,
) -> float:
    """Compute average unique-category diversity normalized by available categories."""
    if not all_results or k <= 0:
        return 0.0
    available_categories = {category for category in item_to_category.values() if category}
    denominator = max(1, min(int(k), len(available_categories)))
    values = []
    for results in all_results:
        categories = {
            item_to_category.get(str(item_id), "")
            for item_id in results[:k]
            if item_to_category.get(str(item_id), "")
        }
        values.append(len(categories) / denominator)
    return float(np.mean(values))


def long_tail_coverage_at_k(
    all_results: list[list[str]],
    long_tail_items: set[str],
    k: int,
) -> float:
    """Compute coverage of long-tail items in retrieved top-k lists."""
    if not long_tail_items:
        return 0.0
    retrieved = {str(item_id) for results in all_results for item_id in results[:k]}
    return float(
        len(retrieved & {str(item_id) for item_id in long_tail_items}) / len(long_tail_items)
    )


def evaluate_multimodal_retriever(
    retriever: Any,
    query_item_pairs: pd.DataFrame,
    items: pd.DataFrame,
    train_interactions: pd.DataFrame,
    k_values: list[int],
    top_k: int,
    cold_start_items: set[str],
    long_tail_items: set[str],
) -> pd.DataFrame:
    """Evaluate multimodal retriever and return per-query rows."""
    del train_interactions
    if query_item_pairs.empty:
        return pd.DataFrame()
    item_to_category = {
        str(row["item_id"]): str(row.get("category", ""))
        for _, row in items[["item_id", "category"]].iterrows()
    }
    all_item_ids = set(items["item_id"].astype(str).tolist())
    rows: list[dict[str, Any]] = []
    all_results: list[list[str]] = []
    split = str(query_item_pairs["split"].iloc[0]) if "split" in query_item_pairs else "unknown"

    for _, query_row in query_item_pairs.sort_values("query_id").iterrows():
        query_text = str(query_row["query_text"])
        target_item_id = str(query_row["target_item_id"])
        start = time.perf_counter()
        results = retriever.search(query_text, top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000.0
        item_ids = [str(row["item_id"]) for row in results]
        all_results.append(item_ids)
        row: dict[str, Any] = {
            "query_id": str(query_row.get("query_id", "")),
            "split": split,
            "target_item_id": target_item_id,
            "retrieved_item_ids": item_ids,
            "latency_ms": latency_ms,
            "is_cold_start_target": target_item_id in cold_start_items,
        }
        for k in k_values:
            row[f"recall_at_{k}"] = recall_at_k(item_ids, target_item_id, k)
            row[f"ndcg_at_{k}"] = ndcg_at_k(item_ids, target_item_id, k)
            row[f"mrr_at_{k}"] = mrr_at_k(item_ids, target_item_id, k)
            row[f"cold_start_recall_at_{k}"] = (
                row[f"recall_at_{k}"] if target_item_id in cold_start_items else np.nan
            )
        rows.append(row)

    per_query = pd.DataFrame(rows)
    for k in k_values:
        per_query[f"catalog_coverage_at_{k}"] = catalog_coverage_at_k(all_results, all_item_ids, k)
        per_query[f"category_diversity_at_{k}"] = category_diversity_at_k(
            all_results,
            item_to_category,
            k,
        )
        per_query[f"long_tail_coverage_at_{k}"] = long_tail_coverage_at_k(
            all_results,
            long_tail_items,
            k,
        )
    return per_query


def summarize_multimodal_metrics(per_query_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-query multimodal metrics into one row."""
    if per_query_results.empty:
        return pd.DataFrame(
            [
                {
                    "method": "unknown",
                    "split": "unknown",
                    "num_queries": 0,
                    "avg_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                }
            ]
        )
    metric_prefixes = (
        "recall_at_",
        "ndcg_at_",
        "mrr_at_",
        "cold_start_recall_at_",
        "long_tail_coverage_at_",
        "catalog_coverage_at_",
        "category_diversity_at_",
    )
    metric_columns = [
        column for column in per_query_results.columns if column.startswith(metric_prefixes)
    ]
    summary: dict[str, Any] = {
        "split": per_query_results["split"].iloc[0],
        "num_queries": int(len(per_query_results)),
        "avg_latency_ms": float(per_query_results["latency_ms"].mean()),
        "p95_latency_ms": float(np.percentile(per_query_results["latency_ms"], 95)),
    }
    for column in sorted(metric_columns):
        if column.startswith(
            ("catalog_coverage_at_", "category_diversity_at_", "long_tail_coverage_at_")
        ):
            summary[column] = float(per_query_results[column].iloc[0])
        else:
            summary[column] = float(per_query_results[column].mean(skipna=True))
    return pd.DataFrame([summary])
