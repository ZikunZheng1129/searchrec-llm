"""Evaluation utilities for sequential recommendation models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.evaluation.recommendation_eval import (
    coverage_at_k,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from src.sequence_models.dataset import SequenceMappings, build_eval_examples
from src.sequence_models.trainer import create_sequence_model


def load_sequence_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> tuple[torch.nn.Module, SequenceMappings, dict[str, Any]]:
    """Load a trained sequence model checkpoint."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    config = checkpoint["config"]
    mappings = checkpoint["mappings"]
    method = str(config.get("sequence", {}).get("method"))
    model = create_sequence_model(method=method, num_items=mappings.num_items, config=config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model, mappings, config


def _pad_seen_ids(seen_ids: list[int], device: torch.device) -> torch.Tensor:
    if not seen_ids:
        return torch.zeros((1, 1), dtype=torch.long, device=device)
    return torch.tensor([seen_ids], dtype=torch.long, device=device)


@torch.no_grad()
def evaluate_sequence_model(
    model: torch.nn.Module,
    train_interactions: pd.DataFrame,
    eval_interactions: pd.DataFrame,
    items: pd.DataFrame,
    mappings: SequenceMappings,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Evaluate a sequence model on next-item examples."""
    device = next(model.parameters()).device
    sequence_config = config.get("sequence", {})
    evaluation_config = config.get("evaluation", {})
    method = str(sequence_config.get("method"))
    split = str(sequence_config.get("eval_split", "test"))
    top_k = int(evaluation_config.get("top_k", 20))
    k_values = [int(k) for k in evaluation_config.get("k_values", [10, 20])]
    exclude_seen = bool(sequence_config.get("exclude_seen", True))
    max_seq_len = int(sequence_config.get("max_seq_len", 20))
    examples = build_eval_examples(
        train_interactions=train_interactions,
        eval_interactions=eval_interactions,
        mappings=mappings,
        max_seq_len=max_seq_len,
    )
    all_item_ids = set(items["item_id"].dropna().astype(str).tolist())
    rows: list[dict[str, Any]] = []
    all_recommendations: dict[str, list[str]] = {}

    for example in examples:
        input_ids = torch.tensor([example["input_ids"]], dtype=torch.long, device=device)
        exclude_ids = _pad_seen_ids(example["seen_ids"], device) if exclude_seen else None
        start_time = time.perf_counter()
        _, recommended_encoded_ids = model.recommend(
            input_ids=input_ids,
            top_k=top_k,
            exclude_ids=exclude_ids,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        recommended_items = [
            mappings.index_to_item_id[int(item_idx)]
            for item_idx in recommended_encoded_ids.squeeze(0).detach().cpu().tolist()
            if int(item_idx) in mappings.index_to_item_id
        ]
        target_items = {
            mappings.index_to_item_id[int(item_idx)]
            for item_idx in example["target_ids"]
            if int(item_idx) in mappings.index_to_item_id
        }
        all_recommendations[example["user_id"]] = recommended_items
        row: dict[str, Any] = {
            "user_id": example["user_id"],
            "method": method,
            "split": split,
            "target_items": sorted(target_items),
            "recommended_item_ids": recommended_items,
            "num_targets": len(target_items),
            "num_recommendations": len(recommended_items),
            "top_k": top_k,
            "latency_ms": latency_ms,
        }
        for k in k_values:
            row[f"hit_rate_at_{k}"] = hit_rate_at_k(recommended_items, target_items, k)
            row[f"recall_at_{k}"] = recall_at_k(recommended_items, target_items, k)
            row[f"ndcg_at_{k}"] = ndcg_at_k(recommended_items, target_items, k)
            row[f"mrr_at_{k}"] = mrr_at_k(recommended_items, target_items, k)
        rows.append(row)

    per_user = pd.DataFrame(rows)
    if per_user.empty:
        return per_user
    for k in k_values:
        per_user[f"coverage_at_{k}"] = coverage_at_k(all_recommendations, all_item_ids, k)
    return per_user


def summarize_sequence_metrics(per_user_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize sequence per-user metrics into one row."""
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
