import pandas as pd

from src.evaluation.retrieval_eval import (
    evaluate_retriever,
    mrr_at_k,
    recall_at_k,
    summarize_retrieval_metrics,
)
from src.retrieval.bm25_retriever import BM25Retriever


class DummyRetriever:
    method_name = "dummy"
    index_backend = "test"

    def search(self, query: str, top_k: int = 50) -> list[dict]:
        if query == "empty":
            return []
        return [
            {"item_id": "item_b", "score": 0.9, "rank": 1},
            {"item_id": "item_a", "score": 0.8, "rank": 2},
        ][:top_k]


def test_recall_at_k_returns_one_when_target_present() -> None:
    assert recall_at_k(["item_a", "item_b"], "item_b", 2) == 1.0


def test_recall_at_k_returns_zero_when_target_absent() -> None:
    assert recall_at_k(["item_a", "item_b"], "item_c", 2) == 0.0


def test_mrr_at_k_computes_reciprocal_rank() -> None:
    assert mrr_at_k(["item_a", "item_b"], "item_b", 2) == 0.5


def test_evaluate_retriever_returns_expected_metric_columns() -> None:
    pairs = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "query_text": "test query",
                "target_item_id": "item_a",
                "split": "test",
            }
        ]
    )

    results = evaluate_retriever(DummyRetriever(), pairs, k_values=[1, 2], top_k=2)

    assert {"recall_at_1", "recall_at_2", "mrr_at_1", "mrr_at_2", "latency_ms"}.issubset(
        results.columns
    )
    assert results.loc[0, "recall_at_2"] == 1.0


def test_query_coverage_is_computed() -> None:
    pairs = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "query_text": "test query",
                "target_item_id": "item_a",
                "split": "test",
            },
            {
                "query_id": "q2",
                "query_text": "empty",
                "target_item_id": "item_z",
                "split": "test",
            },
        ]
    )

    per_query = evaluate_retriever(DummyRetriever(), pairs, k_values=[2], top_k=2)
    summary = summarize_retrieval_metrics(per_query)

    assert summary.loc[0, "query_coverage"] == 0.5


def test_end_to_end_retrieval_evaluation_on_tiny_dataset() -> None:
    items = pd.DataFrame(
        [
            {
                "item_id": "item_a",
                "title": "Wireless Earbuds",
                "category": "Electronics",
                "brand": "Aster",
                "description": "Clear wireless audio.",
            },
            {
                "item_id": "item_b",
                "title": "Face Serum",
                "category": "Beauty",
                "brand": "Luma",
                "description": "Hydrating skincare for dry skin.",
            },
        ]
    )
    pairs = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "query_text": "wireless earbuds",
                "target_item_id": "item_a",
                "split": "test",
            }
        ]
    )
    retriever = BM25Retriever().fit(items)

    per_query = evaluate_retriever(retriever, pairs, k_values=[1], top_k=1)

    assert per_query.loc[0, "recall_at_1"] == 1.0
