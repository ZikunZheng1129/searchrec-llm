from pathlib import Path

import pandas as pd

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import TfidfDenseRetriever
from src.retrieval.faiss_retriever import FaissRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.overlap_rerank_retriever import OverlapRerankRetriever


def _items() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item_id": "item_001",
                "title": "Wireless Earbuds",
                "category": "Electronics",
                "brand": "Aster",
                "description": "Compact earbuds with clear wireless audio.",
            },
            {
                "item_id": "item_002",
                "title": "Trail Running Shoes",
                "category": "Sports & Outdoors",
                "brand": "Stride",
                "description": "Comfortable shoes for running and walking.",
            },
            {
                "item_id": "item_003",
                "title": "Hydrating Face Serum",
                "category": "Beauty",
                "brand": "Luma",
                "description": "Gentle skincare serum for dry skin.",
            },
        ]
    )


def test_bm25_fit_search_returns_top_k_results() -> None:
    retriever = BM25Retriever().fit(_items())

    results = retriever.search("wireless earbuds", top_k=2)

    assert len(results) == 2
    assert {"item_id", "score", "rank"}.issubset(results[0])
    assert results[0]["rank"] == 1


def test_bm25_retrieves_obvious_title_match() -> None:
    retriever = BM25Retriever().fit(_items())

    results = retriever.search("wireless earbuds", top_k=3)

    assert results[0]["item_id"] == "item_001"


def test_dense_retriever_is_deterministic() -> None:
    first = TfidfDenseRetriever().fit(_items()).search("dry skin serum", top_k=3)
    second = TfidfDenseRetriever().fit(_items()).search("dry skin serum", top_k=3)

    assert first == second
    assert first[0]["item_id"] == "item_003"


def test_dense_retriever_handles_unseen_query_tokens() -> None:
    retriever = TfidfDenseRetriever().fit(_items())

    results = retriever.search("zzzz qqqq unseen", top_k=2)

    assert len(results) == 2
    assert all(result["score"] == 0.0 for result in results)


def test_faiss_retriever_numpy_fallback_searches() -> None:
    retriever = FaissRetriever(backend="numpy", fallback_to_numpy=True).fit(_items())

    results = retriever.search("running shoes", top_k=2)

    assert retriever.backend == "numpy"
    assert len(results) == 2
    assert results[0]["item_id"] == "item_002"


def test_hybrid_retriever_returns_component_scores() -> None:
    retriever = HybridRetriever().fit(_items())

    results = retriever.search("wireless electronics", top_k=2)

    assert len(results) == 2
    assert {"bm25_score", "dense_score"}.issubset(results[0])


def test_bm25_save_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "bm25.pkl"
    retriever = BM25Retriever().fit(_items())
    retriever.save(path)

    loaded = BM25Retriever.load(path)

    assert loaded.search("wireless earbuds", top_k=1)[0]["item_id"] == "item_001"


def test_dense_save_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "dense.pkl"
    retriever = TfidfDenseRetriever().fit(_items())
    retriever.save(path)

    loaded = TfidfDenseRetriever.load(path)

    assert loaded.search("dry skin", top_k=1)[0]["item_id"] == "item_003"


def test_empty_query_does_not_crash() -> None:
    retriever = BM25Retriever().fit(_items())

    results = retriever.search("", top_k=2)

    assert len(results) == 2
    assert results[0]["rank"] == 1


def test_bm25_field_weights_can_prioritize_title_over_description() -> None:
    items = pd.DataFrame(
        [
            {
                "item_id": "item_title",
                "title": "Alpha",
                "category": "Test",
                "brand": "A",
                "description": "plain text",
            },
            {
                "item_id": "item_description",
                "title": "Plain",
                "category": "Test",
                "brand": "B",
                "description": "Alpha Alpha Alpha",
            },
        ]
    )
    retriever = BM25Retriever(
        item_text_fields=["title", "description"],
        field_weights={"title": 4.0, "description": 0.1},
        b=0.0,
    ).fit(items)

    results = retriever.search("alpha", top_k=2)

    assert results[0]["item_id"] == "item_title"


def test_overlap_rerank_retriever_adds_general_feature_scores() -> None:
    retriever = OverlapRerankRetriever(
        candidate_pool_size=3,
        bm25_params={"k1": 1.5, "b": 0.75},
        weights={"bm25": 0.1, "title_overlap": 1.0},
    ).fit(_items())

    results = retriever.search("wireless earbuds", top_k=2)

    assert len(results) == 2
    assert results[0]["item_id"] == "item_001"
    assert "bm25_score" in results[0]
    assert "rerank_features" in results[0]
