"""Evaluation metrics for Stage 10 GenRec outputs."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.llm.explanations.hallucination_checker import check_valid_item_ids


def _as_list(value: Any) -> list[str]:
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if pd.isna(value):
        return []
    return [str(value)]


def valid_item_rate(recommended_item_ids: list[str], valid_item_ids: set[str]) -> float:
    """Return fraction of recommendation IDs present in the catalog."""
    return float(check_valid_item_ids(recommended_item_ids, valid_item_ids)["valid_item_rate"])


def hallucination_rate(invalid_item_ids: list[str], output_item_ids: list[str]) -> float:
    """Return invalid emitted IDs divided by output IDs."""
    total = len(output_item_ids) if output_item_ids else len(invalid_item_ids)
    return len(invalid_item_ids) / total if total else 0.0


def recall_at_k(recommended_item_ids: list[str], target_item_id: str, k: int) -> float:
    """Return 1 when the target appears in the top-k recommendations."""
    return float(str(target_item_id) in [str(item_id) for item_id in recommended_item_ids[:k]])


def ndcg_at_k(recommended_item_ids: list[str], target_item_id: str, k: int) -> float:
    """Compute single-positive NDCG@K."""
    target = str(target_item_id)
    for index, item_id in enumerate(recommended_item_ids[:k], start=1):
        if str(item_id) == target:
            return float(1.0 / math.log2(index + 1))
    return 0.0


def mrr_at_k(recommended_item_ids: list[str], target_item_id: str, k: int) -> float:
    """Compute reciprocal rank@K."""
    target = str(target_item_id)
    for index, item_id in enumerate(recommended_item_ids[:k], start=1):
        if str(item_id) == target:
            return float(1.0 / index)
    return 0.0


def evaluate_genrec_outputs(
    outputs: pd.DataFrame,
    query_item_pairs: pd.DataFrame,
    item_metadata: pd.DataFrame,
    k_values: list[int],
) -> pd.DataFrame:
    """Evaluate GenRec/reranking outputs per query."""
    if outputs.empty:
        return pd.DataFrame()
    targets = {
        str(row["query_id"]): str(row["target_item_id"]) for _, row in query_item_pairs.iterrows()
    }
    valid_catalog = set(item_metadata["item_id"].astype(str))
    rows = []
    for _, row in outputs.iterrows():
        query_id = str(row.get("query_id", ""))
        recommended = _as_list(row.get("recommended_item_ids", []))
        row_invalid = _as_list(row.get("invalid_item_ids", []))
        catalog_invalid = [item_id for item_id in recommended if str(item_id) not in valid_catalog]
        invalid = list(dict.fromkeys(row_invalid + catalog_invalid))
        target = str(row.get("target_item_id", targets.get(query_id, "")))
        result = {
            "query_id": query_id,
            "method": str(row.get("method", "")),
            "provider": str(row.get("provider", "")),
            "model": str(row.get("model", "")),
            "split": str(row.get("split", "")),
            "valid_item_rate": valid_item_rate(recommended, valid_catalog),
            "hallucination_rate": hallucination_rate(invalid, recommended),
            "output_parse_success": bool(row.get("parse_success", False)),
            "schema_valid": bool(row.get("schema_valid", False)),
            "used_fallback": bool(row.get("used_fallback", False)),
            "latency_ms": float(row.get("latency_ms", 0.0) or 0.0),
            "estimated_cost_usd": float(row.get("estimated_cost_usd", 0.0) or 0.0),
        }
        for k in k_values:
            result[f"recall_at_{k}"] = recall_at_k(recommended, target, int(k))
            result[f"ndcg_at_{k}"] = ndcg_at_k(recommended, target, int(k))
            result[f"mrr_at_{k}"] = mrr_at_k(recommended, target, int(k))
        rows.append(result)
    return pd.DataFrame(rows)


def evaluate_explanations(explanations: pd.DataFrame) -> pd.DataFrame:
    """Evaluate explanation rows for parse and faithfulness metrics."""
    if explanations.empty:
        return pd.DataFrame()
    rows = []
    for _, row in explanations.iterrows():
        rows.append(
            {
                "query_id": str(row.get("query_id", "")),
                "method": str(row.get("method", "")),
                "provider": str(row.get("provider", "")),
                "model": str(row.get("model", "")),
                "split": str(row.get("split", "")),
                "valid_item_rate": np.nan,
                "hallucination_rate": np.nan,
                "output_parse_success": bool(row.get("parse_success", False)),
                "schema_valid": bool(row.get("schema_valid", False)),
                "used_fallback": bool(row.get("used_fallback", False)),
                "explanation_faithful": bool(row.get("explanation_faithful", False)),
                "latency_ms": float(row.get("latency_ms", 0.0) or 0.0),
                "estimated_cost_usd": float(row.get("estimated_cost_usd", 0.0) or 0.0),
                "recall_at_10": np.nan,
                "ndcg_at_10": np.nan,
                "mrr_at_10": np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_genrec_metrics(per_query_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize GenRec metrics by method/provider/model/split."""
    if per_query_results.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["method", "provider", "model", "split"]
    for keys, group in per_query_results.groupby(group_cols, dropna=False, sort=True):
        method, provider, model, split = keys
        num_queries = int(len(group))
        cost = float(group["estimated_cost_usd"].sum()) if "estimated_cost_usd" in group else 0.0
        row = {
            "stage": "genrec",
            "method": method,
            "provider": provider,
            "model": model,
            "split": split,
            "num_queries": num_queries,
            "valid_item_rate": float(group["valid_item_rate"].mean(skipna=True)),
            "hallucination_rate": float(group["hallucination_rate"].mean(skipna=True)),
            "output_parse_success_rate": float(group["output_parse_success"].mean()),
            "schema_valid_rate": float(group["schema_valid"].mean()),
            "fallback_rate": float(group["used_fallback"].mean()),
            "avg_latency_ms": float(group["latency_ms"].mean()),
            "p95_latency_ms": float(group["latency_ms"].quantile(0.95)),
            "estimated_cost_per_1000_queries": cost / max(num_queries, 1) * 1000.0,
        }
        for column in ["recall_at_10", "ndcg_at_10", "mrr_at_10"]:
            row[column] = float(group[column].mean(skipna=True)) if column in group else np.nan
        if "explanation_faithful" in group:
            row["explanation_faithfulness"] = float(group["explanation_faithful"].mean(skipna=True))
        else:
            row["explanation_faithfulness"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)
