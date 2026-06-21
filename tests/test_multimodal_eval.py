from __future__ import annotations

import math

import pandas as pd

from src.evaluation.multimodal_eval import (
    catalog_coverage_at_k,
    category_diversity_at_k,
    evaluate_multimodal_retriever,
    long_tail_coverage_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from src.multimodal.multimodal_retriever import MultimodalRetriever
from src.multimodal.text_encoder import TextItemEncoder


def test_multimodal_metric_correctness() -> None:
    results = ["i2", "i1"]
    assert recall_at_k(results, "i1", 2) == 1.0
    assert mrr_at_k(results, "i1", 2) == 0.5
    assert 0.0 < ndcg_at_k(results, "i1", 2) < 1.0
    assert catalog_coverage_at_k([results], {"i1", "i2", "i3"}, 2) == 2 / 3
    assert long_tail_coverage_at_k([results], {"i1"}, 2) == 1.0
    diversity = category_diversity_at_k([results], {"i1": "A", "i2": "B"}, 2)
    assert diversity == 1.0


def test_evaluate_multimodal_retriever_columns_and_nan_cold_start() -> None:
    items = pd.DataFrame(
        [
            {"item_id": "i1", "title": "Earbuds", "category": "Electronics"},
            {"item_id": "i2", "title": "Mat", "category": "Sports"},
        ]
    )
    queries = pd.DataFrame(
        [{"query_id": "q1", "query_text": "earbuds", "target_item_id": "i1", "split": "test"}]
    )
    encoder = TextItemEncoder(fields=["title", "category"]).fit(items)
    retriever = MultimodalRetriever().fit(items, encoder.transform(items), encoder)
    per_query = evaluate_multimodal_retriever(
        retriever,
        queries,
        items,
        pd.DataFrame(),
        [1, 2],
        2,
        cold_start_items=set(),
        long_tail_items={"i2"},
    )
    assert {"recall_at_1", "ndcg_at_2", "catalog_coverage_at_1"}.issubset(per_query.columns)
    assert math.isnan(per_query["cold_start_recall_at_1"].iloc[0])
