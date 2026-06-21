from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.evaluation.genrec_eval import (
    evaluate_explanations,
    evaluate_genrec_outputs,
    hallucination_rate,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    summarize_genrec_metrics,
    valid_item_rate,
)


def test_basic_genrec_metrics():
    assert valid_item_rate(["a", "x"], {"a"}) == 0.5
    assert hallucination_rate(["x"], ["a"]) == 1.0
    assert recall_at_k(["a"], "a", 10) == 1.0
    assert ndcg_at_k(["x", "a"], "a", 10) == 1 / math.log2(3)
    assert mrr_at_k(["x", "a"], "a", 10) == 0.5


def test_evaluate_genrec_outputs_and_summary_columns():
    outputs = pd.DataFrame(
        {
            "query_id": ["q1"],
            "method": ["candidate_constrained_generation"],
            "provider": ["mock"],
            "model": ["mock"],
            "split": ["test"],
            "recommended_item_ids": [["item_1"]],
            "invalid_item_ids": [[]],
            "parse_success": [True],
            "schema_valid": [True],
            "used_fallback": [False],
            "latency_ms": [1.0],
            "estimated_cost_usd": [0.0],
        }
    )
    pairs = pd.DataFrame({"query_id": ["q1"], "target_item_id": ["item_1"]})
    items = pd.DataFrame({"item_id": ["item_1"]})
    per_query = evaluate_genrec_outputs(outputs, pairs, items, [10])
    summary = summarize_genrec_metrics(per_query)
    assert per_query["ndcg_at_10"].iloc[0] == 1.0
    assert summary["valid_item_rate"].iloc[0] == 1.0
    assert "estimated_cost_per_1000_queries" in summary.columns


def test_explanation_faithfulness_aggregation_and_undefined_metrics():
    explanations = pd.DataFrame(
        {
            "query_id": ["q1"],
            "method": ["template_explanation"],
            "provider": ["template"],
            "model": ["template"],
            "split": ["test"],
            "parse_success": [True],
            "schema_valid": [True],
            "used_fallback": [False],
            "explanation_faithful": [True],
            "latency_ms": [0.0],
            "estimated_cost_usd": [0.0],
        }
    )
    per_query = evaluate_explanations(explanations)
    summary = summarize_genrec_metrics(per_query)
    assert summary["explanation_faithfulness"].iloc[0] == 1.0
    assert np.isnan(summary["ndcg_at_10"].iloc[0])
