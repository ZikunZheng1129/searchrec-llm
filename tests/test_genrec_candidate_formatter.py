from __future__ import annotations

import pandas as pd

from src.llm.genrec.candidate_formatter import (
    build_candidate_lookup,
    format_candidates_for_llm,
    load_candidate_source,
    select_top_candidates_for_query,
)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q1", "q1"],
            "candidate_item_id": ["item_b", "item_a", "item_b", "item_c"],
            "multimodal_score": [0.2, 0.9, 0.8, 0.1],
            "multimodal_rank": [2, 1, 3, 4],
            "category": ["Beauty", "Beauty", "Beauty", "Home"],
            "brand": ["B", "A", "B", "C"],
            "price": [10.0, 20.0, 10.0, 5.0],
            "avg_rating": [4.0, 4.8, 4.0, 3.5],
            "rating_count": [10, 20, 10, 5],
        }
    )


def _items() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["item_a", "item_b"],
            "title": ["Alpha Serum", "Beta Cream"],
            "category": ["Beauty", "Beauty"],
            "brand": ["A", "B"],
            "price": [20.0, 10.0],
            "avg_rating": [4.8, 4.0],
            "rating_count": [20, 10],
            "description": ["Brightening serum", "Daily cream"],
        }
    )


def test_candidate_source_loading_prefers_multimodal(tmp_path):
    ranking_path = tmp_path / "ranking.parquet"
    multimodal_path = tmp_path / "ranking_multimodal.parquet"
    pd.DataFrame({"query_id": ["q1"], "candidate_item_id": ["base"]}).to_parquet(ranking_path)
    pd.DataFrame({"query_id": ["q1"], "candidate_item_id": ["multi"]}).to_parquet(multimodal_path)
    loaded = load_candidate_source(
        {
            "input": {
                "ranking_candidates_path": str(ranking_path),
                "ranking_candidates_multimodal_path": str(multimodal_path),
            },
            "genrec": {"candidate_source": "auto"},
        }
    )
    assert loaded["candidate_item_id"].tolist() == ["multi"]


def test_top_candidate_selection_is_deterministic_and_deduped():
    selected = select_top_candidates_for_query(_candidates(), "q1", top_n=3)
    assert selected["candidate_item_id"].tolist() == ["item_a", "item_b", "item_c"]


def test_candidate_formatting_includes_required_fields_and_missing_metadata():
    selected = select_top_candidates_for_query(_candidates(), "q1", top_n=3)
    formatted = format_candidates_for_llm(selected, _items(), max_candidates=3)
    assert formatted[0]["item_id"] == "item_a"
    assert formatted[0]["title"] == "Alpha Serum"
    assert formatted[-1]["item_id"] == "item_c"
    assert set(
        [
            "item_id",
            "title",
            "category",
            "brand",
            "price",
            "avg_rating",
            "rating_count",
            "score",
            "rank",
            "evidence_text",
        ]
    ).issubset(formatted[0])
    assert build_candidate_lookup(formatted)["item_a"]["brand"] == "A"
