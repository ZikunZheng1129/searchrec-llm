from __future__ import annotations

import math

import pandas as pd

from src.evaluation.ranking_eval import (
    auc_score,
    evaluate_ranking_predictions,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    summarize_ranking_metrics,
)


def test_ranking_metric_correctness() -> None:
    labels = [0, 1, 0]
    assert mrr_at_k(labels, 3) == 0.5
    assert precision_at_k(labels, 2) == 0.5
    assert recall_at_k(labels, total_relevant=1, k=2) == 1.0
    assert 0.0 < ndcg_at_k(labels, 3) < 1.0


def test_auc_score_normal_and_undefined() -> None:
    assert auc_score([1, 0], [0.9, 0.1]) == 1.0
    assert math.isnan(auc_score([1, 1], [0.9, 0.1]))


def test_evaluate_ranking_predictions_and_summary() -> None:
    scored = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "candidate_item_id": "i1",
                "label": 1,
                "score": 0.9,
                "split": "test",
            },
            {
                "query_id": "q1",
                "candidate_item_id": "i2",
                "label": 0,
                "score": 0.1,
                "split": "test",
            },
            {
                "query_id": "q2",
                "candidate_item_id": "i3",
                "label": 0,
                "score": 0.8,
                "split": "test",
            },
            {
                "query_id": "q2",
                "candidate_item_id": "i4",
                "label": 0,
                "score": 0.2,
                "split": "test",
            },
        ]
    )
    per_query = evaluate_ranking_predictions(scored, [1, 2], "score")
    assert {"ndcg_at_1", "mrr_at_2", "precision_at_1", "recall_at_2"}.issubset(per_query.columns)
    assert per_query["candidate_covered"].mean() == 0.5
    summary = summarize_ranking_metrics(per_query)
    assert summary["num_queries"].iloc[0] == 2
    assert summary["candidate_coverage"].iloc[0] == 0.5
