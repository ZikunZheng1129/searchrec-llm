"""Retrieval evaluation metrics."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd


def recall_at_k(results: list[str], target_item_id: str, k: int) -> float:
    """Return 1.0 if the target item is present in the top-k results."""
    return 1.0 if str(target_item_id) in [str(item_id) for item_id in results[:k]] else 0.0


def mrr_at_k(results: list[str], target_item_id: str, k: int) -> float:
    """Compute reciprocal rank if the target item is present in the top-k results."""
    target = str(target_item_id)
    for rank, item_id in enumerate(results[:k], start=1):
        if str(item_id) == target:
            return 1.0 / rank
    return 0.0


def _retriever_method_name(retriever: Any) -> str:
    return str(getattr(retriever, "method_name", retriever.__class__.__name__.lower()))


def _retriever_backend(retriever: Any) -> str:
    return str(getattr(retriever, "backend", getattr(retriever, "index_backend", "local")))


def evaluate_retriever(
    retriever: Any,
    query_item_pairs: pd.DataFrame,
    k_values: list[int],
    top_k: int,
) -> pd.DataFrame:
    """Evaluate a retriever on query-item positive pairs and return per-query metrics."""
    rows: list[dict[str, Any]] = []
    method = _retriever_method_name(retriever)
    backend = _retriever_backend(retriever)

    for _, row in query_item_pairs.reset_index(drop=True).iterrows():
        query_text = str(row["query_text"])
        target_item_id = str(row["target_item_id"])
        start_time = time.perf_counter()
        results = retriever.search(query_text, top_k=top_k)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        retrieved_item_ids = [str(result["item_id"]) for result in results]

        result_row: dict[str, Any] = {
            "query_id": row.get("query_id"),
            "query_text": query_text,
            "target_item_id": target_item_id,
            "split": row.get("split", "unknown"),
            "method": method,
            "index_backend": backend,
            "top_k": int(top_k),
            "retrieved_item_ids": retrieved_item_ids,
            "num_results": len(results),
            "has_results": len(results) > 0,
            "latency_ms": latency_ms,
        }
        for k in k_values:
            result_row[f"recall_at_{k}"] = recall_at_k(retrieved_item_ids, target_item_id, k)
            result_row[f"mrr_at_{k}"] = mrr_at_k(retrieved_item_ids, target_item_id, k)
        rows.append(result_row)

    return pd.DataFrame(rows)


def summarize_retrieval_metrics(per_query_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-query retrieval metrics into one row."""
    metric_columns = [
        column
        for column in per_query_results.columns
        if column.startswith("recall_at_") or column.startswith("mrr_at_")
    ]
    if per_query_results.empty:
        base = {
            "method": "unknown",
            "split": "unknown",
            "num_queries": 0,
            "top_k": 0,
            "query_coverage": 0.0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "index_backend": "unknown",
        }
        return pd.DataFrame([base])

    summary = {
        "method": per_query_results["method"].iloc[0],
        "split": per_query_results["split"].iloc[0],
        "num_queries": int(len(per_query_results)),
        "top_k": int(per_query_results["top_k"].iloc[0]),
        "query_coverage": float(per_query_results["has_results"].mean()),
        "avg_latency_ms": float(per_query_results["latency_ms"].mean()),
        "p95_latency_ms": float(np.percentile(per_query_results["latency_ms"], 95)),
        "index_backend": per_query_results["index_backend"].iloc[0],
    }
    for column in sorted(metric_columns):
        summary[column] = float(per_query_results[column].mean())
    return pd.DataFrame([summary])
