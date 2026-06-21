"""Feature generation for local query-item ranking."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import TfidfDenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.text_utils import build_item_text, tokenize

RANKING_CANDIDATE_COLUMNS = [
    "query_id",
    "query_text",
    "split",
    "target_item_id",
    "candidate_item_id",
    "label",
    "candidate_source",
    "bm25_score",
    "dense_score",
    "hybrid_score",
    "bm25_rank",
    "dense_rank",
    "hybrid_rank",
    "category_match",
    "brand_match",
    "title_token_overlap",
    "description_token_overlap",
    "query_length",
    "item_title_length",
    "item_description_length",
    "item_popularity_count",
    "item_event_weight_sum",
    "item_purchase_count",
    "item_add_to_cart_count",
    "avg_rating",
    "rating_count",
    "log_rating_count",
    "price",
    "price_bucket",
    "authority_score",
    "conversion_proxy",
    "cold_start_score",
    "diversity_category",
    "category",
    "brand",
]


def build_candidate_item_text(items: pd.DataFrame) -> pd.DataFrame:
    """Add a deterministic item_text column used by local rankers."""
    output = items.copy()
    output["item_id"] = output["item_id"].astype(str)
    output["item_text"] = [build_item_text(row) for _, row in output.iterrows()]
    return output


def compute_item_popularity_features(train_interactions: pd.DataFrame) -> pd.DataFrame:
    """Compute item-level interaction counts from the train split."""
    if train_interactions.empty:
        return pd.DataFrame(
            columns=[
                "item_id",
                "item_popularity_count",
                "item_event_weight_sum",
                "item_purchase_count",
                "item_add_to_cart_count",
            ]
        )

    interactions = train_interactions.copy()
    interactions["item_id"] = interactions["item_id"].astype(str)
    if "event_type" not in interactions.columns:
        interactions["event_type"] = ""
    interactions["event_type"] = interactions["event_type"].astype(str)
    if "event_weight" not in interactions.columns:
        interactions["event_weight"] = 0.0
    interactions["event_weight"] = pd.to_numeric(
        interactions["event_weight"],
        errors="coerce",
    ).fillna(0.0)

    grouped = interactions.groupby("item_id", sort=True)
    output = grouped.size().rename("item_popularity_count").reset_index()
    output["item_event_weight_sum"] = grouped["event_weight"].sum().values
    output["item_purchase_count"] = (
        grouped["event_type"].apply(lambda values: int((values == "purchase").sum())).values
    )
    output["item_add_to_cart_count"] = (
        grouped["event_type"].apply(lambda values: int((values == "add_to_cart").sum())).values
    )
    return output


def compute_conversion_proxy(train_interactions: pd.DataFrame) -> pd.DataFrame:
    """Compute a deterministic conversion proxy from synthetic event types."""
    popularity = compute_item_popularity_features(train_interactions)
    if popularity.empty:
        return pd.DataFrame(columns=["item_id", "conversion_proxy"])

    denominator = popularity["item_popularity_count"].clip(lower=1)
    conversion = (
        popularity["item_purchase_count"] + 0.5 * popularity["item_add_to_cart_count"]
    ) / denominator
    return pd.DataFrame(
        {
            "item_id": popularity["item_id"],
            "conversion_proxy": conversion.astype(float),
        }
    )


def compute_price_bucket(price: float) -> str:
    """Bucket a numeric price into a stable coarse category."""
    value = float(price) if pd.notna(price) else 0.0
    if value < 50.0:
        return "low"
    if value < 100.0:
        return "mid"
    if value < 200.0:
        return "high"
    return "premium"


def compute_authority_score(avg_rating: float, rating_count: int) -> float:
    """Compute a bounded authority proxy from item rating metadata."""
    rating = float(avg_rating) if pd.notna(avg_rating) else 0.0
    count = max(0, int(rating_count) if pd.notna(rating_count) else 0)
    rating_component = max(0.0, min(rating / 5.0, 1.0))
    count_component = math.log1p(count) / math.log1p(5000)
    return float(rating_component * min(count_component, 1.0))


def compute_business_proxy_features(
    train_interactions: pd.DataFrame,
    items: pd.DataFrame,
) -> pd.DataFrame:
    """Build item-level proxy features from synthetic interactions and metadata."""
    item_ids = items[["item_id"]].copy()
    item_ids["item_id"] = item_ids["item_id"].astype(str)
    popularity = compute_item_popularity_features(train_interactions)
    conversion = compute_conversion_proxy(train_interactions)
    output = item_ids.merge(popularity, on="item_id", how="left").merge(
        conversion,
        on="item_id",
        how="left",
    )
    fill_values = {
        "item_popularity_count": 0,
        "item_event_weight_sum": 0.0,
        "item_purchase_count": 0,
        "item_add_to_cart_count": 0,
        "conversion_proxy": 0.0,
    }
    return output.fillna(fill_values)


def _overlap_ratio(query_tokens: list[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    text_tokens = set(tokenize(text))
    if not text_tokens:
        return 0.0
    return len(set(query_tokens) & text_tokens) / len(set(query_tokens))


def compute_text_match_features(query_text: str, item_row: pd.Series) -> dict[str, float]:
    """Compute simple lexical overlap features for a query-item pair."""
    query_tokens = tokenize(query_text)
    title = str(item_row.get("title", ""))
    description = str(item_row.get("description", ""))
    category = str(item_row.get("category", ""))
    brand = str(item_row.get("brand", ""))
    query_token_set = set(query_tokens)
    return {
        "category_match": float(bool(query_token_set & set(tokenize(category)))),
        "brand_match": float(bool(query_token_set & set(tokenize(brand)))),
        "title_token_overlap": float(_overlap_ratio(query_tokens, title)),
        "description_token_overlap": float(_overlap_ratio(query_tokens, description)),
        "query_length": float(len(query_tokens)),
        "item_title_length": float(len(tokenize(title))),
        "item_description_length": float(len(tokenize(description))),
    }


def _ranking_maps(results: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, int]]:
    scores = {str(row["item_id"]): float(row["score"]) for row in results}
    ranks = {str(row["item_id"]): int(row["rank"]) for row in results}
    return scores, ranks


def _candidate_sources(
    item_id: str,
    target_item_id: str,
    bm25_ranks: dict[str, int],
    dense_ranks: dict[str, int],
    hybrid_ranks: dict[str, int],
    forced_positive: bool,
    random_negative: bool,
) -> str:
    sources = []
    if item_id in bm25_ranks:
        sources.append("bm25")
    if item_id in dense_ranks:
        sources.append("dense")
    if item_id in hybrid_ranks:
        sources.append("hybrid")
    if forced_positive and item_id == target_item_id:
        sources.append("positive_forced")
    if random_negative:
        sources.append("random_negative")
    return "+".join(sources) if sources else "unknown"


def _select_candidates(
    target_item_id: str,
    item_ids: list[str],
    bm25_ranks: dict[str, int],
    dense_ranks: dict[str, int],
    hybrid_ranks: dict[str, int],
    candidate_pool_size: int,
    random_negatives_per_query: int,
    ensure_positive_candidate: bool,
    rng: np.random.Generator,
) -> tuple[list[str], set[str], set[str]]:
    ordered_candidates: list[str] = []
    for ranks in (hybrid_ranks, bm25_ranks, dense_ranks):
        for item_id, _ in sorted(ranks.items(), key=lambda row: (row[1], row[0])):
            if item_id not in ordered_candidates:
                ordered_candidates.append(item_id)
            if len(ordered_candidates) >= candidate_pool_size:
                break
        if len(ordered_candidates) >= candidate_pool_size:
            break

    forced_positive: set[str] = set()
    if ensure_positive_candidate and target_item_id not in ordered_candidates:
        ordered_candidates.append(target_item_id)
        forced_positive.add(target_item_id)

    random_added: set[str] = set()
    if len(ordered_candidates) < candidate_pool_size and random_negatives_per_query > 0:
        available = [
            item_id
            for item_id in item_ids
            if item_id != target_item_id and item_id not in ordered_candidates
        ]
        sample_size = min(
            random_negatives_per_query,
            candidate_pool_size - len(ordered_candidates),
            len(available),
        )
        if sample_size > 0:
            sampled = sorted(rng.choice(available, size=sample_size, replace=False).tolist())
            ordered_candidates.extend(sampled)
            random_added.update(sampled)

    return ordered_candidates, forced_positive, random_added


def build_ranking_features(
    query_item_pairs: pd.DataFrame,
    items: pd.DataFrame,
    train_interactions: pd.DataFrame,
    candidate_pool_size: int,
    random_negatives_per_query: int,
    seed: int,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build a query-item candidate table for ranking experiments."""
    if query_item_pairs.empty:
        return pd.DataFrame(columns=RANKING_CANDIDATE_COLUMNS)
    required = {"query_text", "target_item_id"}
    missing = sorted(required - set(query_item_pairs.columns))
    if missing:
        raise ValueError(f"query_item_pairs missing required columns: {missing}")
    if "item_id" not in items.columns:
        raise ValueError("items must contain an item_id column")

    rng = np.random.default_rng(seed)
    item_table = build_candidate_item_text(items).copy()
    item_table["item_id"] = item_table["item_id"].astype(str)
    item_table = item_table.sort_values("item_id").reset_index(drop=True)
    item_lookup = {str(row["item_id"]): row for _, row in item_table.iterrows()}
    item_ids = sorted(item_lookup)
    proxy_features = compute_business_proxy_features(train_interactions, item_table)
    proxy_lookup = {str(row["item_id"]): row for _, row in proxy_features.iterrows()}

    retrieval_config = config.get("retrieval", {})
    bm25 = BM25Retriever(**retrieval_config.get("bm25", {})).fit(item_table)
    dense_config = dict(retrieval_config.get("dense", {}))
    dense_config.pop("vectorizer", None)
    dense = TfidfDenseRetriever(**dense_config).fit(item_table)
    hybrid = HybridRetriever(
        bm25_params=retrieval_config.get("bm25", {}),
        dense_params=dense_config,
        **retrieval_config.get("hybrid", {}),
    ).fit(item_table)

    full_count = len(item_ids)
    rows: list[dict[str, Any]] = []
    pairs = query_item_pairs.copy()
    pairs["query_text"] = pairs["query_text"].astype(str)
    pairs["target_item_id"] = pairs["target_item_id"].astype(str)
    if "query_id" not in pairs.columns:
        pairs["query_id"] = [f"query_{index:05d}" for index in range(len(pairs))]
    pairs["query_id"] = pairs["query_id"].astype(str)
    pairs["split"] = pairs.get("split", "unknown")
    pairs = pairs.sort_values(["split", "query_id", "target_item_id"]).reset_index(drop=True)

    ensure_positive = bool(
        config.get("candidate_generation", {}).get("ensure_positive_candidate", True)
    )

    for _, query_row in pairs.iterrows():
        query_id = str(query_row["query_id"])
        query_text = str(query_row["query_text"])
        target_item_id = str(query_row["target_item_id"])
        bm25_scores, bm25_ranks = _ranking_maps(bm25.search(query_text, top_k=full_count))
        dense_scores, dense_ranks = _ranking_maps(dense.search(query_text, top_k=full_count))
        hybrid_results = hybrid.search(query_text, top_k=full_count)
        hybrid_scores = {str(row["item_id"]): float(row["score"]) for row in hybrid_results}
        hybrid_ranks = {str(row["item_id"]): int(row["rank"]) for row in hybrid_results}

        candidates, forced_positive, random_added = _select_candidates(
            target_item_id=target_item_id,
            item_ids=item_ids,
            bm25_ranks=bm25_ranks,
            dense_ranks=dense_ranks,
            hybrid_ranks=hybrid_ranks,
            candidate_pool_size=int(candidate_pool_size),
            random_negatives_per_query=int(random_negatives_per_query),
            ensure_positive_candidate=ensure_positive,
            rng=rng,
        )

        seen: set[str] = set()
        for candidate_item_id in candidates:
            if candidate_item_id in seen:
                continue
            seen.add(candidate_item_id)
            item_row = item_lookup[candidate_item_id]
            proxy_row = proxy_lookup.get(candidate_item_id, pd.Series(dtype=object))
            text_features = compute_text_match_features(query_text, item_row)
            rating_count = int(item_row.get("rating_count", 0) or 0)
            avg_rating = float(item_row.get("avg_rating", 0.0) or 0.0)
            popularity_count = float(proxy_row.get("item_popularity_count", 0.0) or 0.0)
            row = {
                "query_id": query_id,
                "query_text": query_text,
                "split": str(query_row.get("split", "unknown")),
                "target_item_id": target_item_id,
                "candidate_item_id": candidate_item_id,
                "label": int(candidate_item_id == target_item_id),
                "candidate_source": _candidate_sources(
                    candidate_item_id,
                    target_item_id,
                    bm25_ranks,
                    dense_ranks,
                    hybrid_ranks,
                    candidate_item_id in forced_positive,
                    candidate_item_id in random_added,
                ),
                "bm25_score": float(bm25_scores.get(candidate_item_id, 0.0)),
                "dense_score": float(dense_scores.get(candidate_item_id, 0.0)),
                "hybrid_score": float(hybrid_scores.get(candidate_item_id, 0.0)),
                "bm25_rank": int(bm25_ranks.get(candidate_item_id, full_count + 1)),
                "dense_rank": int(dense_ranks.get(candidate_item_id, full_count + 1)),
                "hybrid_rank": int(hybrid_ranks.get(candidate_item_id, full_count + 1)),
                **text_features,
                "item_popularity_count": popularity_count,
                "item_event_weight_sum": float(proxy_row.get("item_event_weight_sum", 0.0) or 0.0),
                "item_purchase_count": float(proxy_row.get("item_purchase_count", 0.0) or 0.0),
                "item_add_to_cart_count": float(
                    proxy_row.get("item_add_to_cart_count", 0.0) or 0.0
                ),
                "avg_rating": avg_rating,
                "rating_count": rating_count,
                "log_rating_count": float(math.log1p(max(0, rating_count))),
                "price": float(item_row.get("price", 0.0) or 0.0),
                "price_bucket": compute_price_bucket(float(item_row.get("price", 0.0) or 0.0)),
                "authority_score": compute_authority_score(avg_rating, rating_count),
                "conversion_proxy": float(proxy_row.get("conversion_proxy", 0.0) or 0.0),
                "cold_start_score": float(1.0 / (1.0 + math.log1p(popularity_count))),
                "diversity_category": str(item_row.get("category", "")),
                "category": str(item_row.get("category", "")),
                "brand": str(item_row.get("brand", "")),
            }
            rows.append(row)

    output = pd.DataFrame(rows)
    for column in RANKING_CANDIDATE_COLUMNS:
        if column not in output.columns:
            output[column] = ""
    numeric_columns = [
        column
        for column in output.columns
        if column not in {"query_id", "query_text", "split", "target_item_id", "candidate_item_id"}
        and pd.api.types.is_numeric_dtype(output[column])
    ]
    output[numeric_columns] = output[numeric_columns].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return (
        output[RANKING_CANDIDATE_COLUMNS]
        .sort_values(
            ["split", "query_id", "candidate_item_id"],
        )
        .reset_index(drop=True)
    )
