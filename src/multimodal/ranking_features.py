"""Optional ranking feature augmentation with multimodal scores."""

from __future__ import annotations

import pandas as pd

from src.multimodal.multimodal_retriever import MultimodalRetriever


def add_multimodal_scores_to_ranking_candidates(
    ranking_candidates: pd.DataFrame,
    query_item_pairs: pd.DataFrame,
    items: pd.DataFrame,
    retriever: MultimodalRetriever,
    score_column: str = "multimodal_score",
    rank_column: str = "multimodal_rank",
) -> pd.DataFrame:
    """Add multimodal score and rank columns while preserving row count."""
    del items
    output = ranking_candidates.copy()
    query_lookup = {
        str(row["query_id"]): str(row["query_text"])
        for _, row in query_item_pairs[["query_id", "query_text"]].drop_duplicates().iterrows()
    }
    score_maps: dict[str, dict[str, tuple[float, int]]] = {}
    for query_id, query_text in query_lookup.items():
        results = retriever.search(query_text, top_k=len(retriever.item_ids))
        score_maps[query_id] = {
            str(row["item_id"]): (float(row["score"]), int(row["rank"])) for row in results
        }
    output[score_column] = 0.0
    output[rank_column] = len(retriever.item_ids) + 1
    for index, row in output.iterrows():
        query_id = str(row["query_id"])
        item_id = str(row["candidate_item_id"])
        score_rank = score_maps.get(query_id, {}).get(item_id)
        if score_rank is not None:
            output.at[index, score_column] = score_rank[0]
            output.at[index, rank_column] = score_rank[1]
    return output
