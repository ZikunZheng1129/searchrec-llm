import pandas as pd

from src.recommendation.itemcf import ItemCFRecommender
from src.recommendation.matrix_factorization import MatrixFactorizationRecommender
from src.recommendation.popularity import PopularityRecommender
from src.recommendation.user_history_embedding import UserHistoryEmbeddingRecommender


def _items() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item_id": "item_001",
                "title": "Wireless Earbuds",
                "category": "Electronics",
                "brand": "Aster",
                "description": "Compact wireless audio.",
            },
            {
                "item_id": "item_002",
                "title": "Trail Running Shoes",
                "category": "Sports & Outdoors",
                "brand": "Stride",
                "description": "Comfortable shoes for running.",
            },
            {
                "item_id": "item_003",
                "title": "Hydrating Face Serum",
                "category": "Beauty",
                "brand": "Luma",
                "description": "Skincare serum for dry skin.",
            },
            {
                "item_id": "item_004",
                "title": "Travel Phone Stand",
                "category": "Electronics",
                "brand": "Orbit",
                "description": "Foldable stand for travel.",
            },
        ]
    )


def _train() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "user_id": "user_1",
                "item_id": "item_001",
                "event_weight": 4.0,
                "rating": 5.0,
            },
            {
                "user_id": "user_1",
                "item_id": "item_004",
                "event_weight": 2.0,
                "rating": 4.0,
            },
            {
                "user_id": "user_2",
                "item_id": "item_002",
                "event_weight": 3.0,
                "rating": 4.0,
            },
            {
                "user_id": "user_2",
                "item_id": "item_001",
                "event_weight": 1.0,
                "rating": 3.0,
            },
            {
                "user_id": "user_3",
                "item_id": "item_003",
                "event_weight": 5.0,
                "rating": 5.0,
            },
        ]
    )


def test_popularity_recommender_schema() -> None:
    recommender = PopularityRecommender().fit(_train(), _items())

    results = recommender.recommend("user_1", top_k=2, exclude_seen=False)

    assert len(results) == 2
    assert {"item_id", "score", "rank"}.issubset(results[0])


def test_popularity_excludes_seen_items() -> None:
    recommender = PopularityRecommender().fit(_train(), _items())

    results = recommender.recommend("user_1", top_k=4, exclude_seen=True)
    item_ids = [result["item_id"] for result in results]

    assert "item_001" not in item_ids
    assert "item_004" not in item_ids


def test_itemcf_returns_recommendations_for_known_user() -> None:
    recommender = ItemCFRecommender().fit(_train(), _items())

    results = recommender.recommend("user_1", top_k=2)

    assert len(results) == 2


def test_itemcf_falls_back_for_unknown_user() -> None:
    recommender = ItemCFRecommender().fit(_train(), _items())

    results = recommender.recommend("new_user", top_k=2)

    assert len(results) == 2


def test_matrix_factorization_trains_and_recommends() -> None:
    recommender = MatrixFactorizationRecommender(
        factors=4,
        epochs=3,
        negative_samples=1,
        seed=7,
    ).fit(_train(), _items())

    results = recommender.recommend("user_1", top_k=2)

    assert len(results) == 2


def test_matrix_factorization_is_deterministic() -> None:
    first = MatrixFactorizationRecommender(
        factors=4,
        epochs=3,
        negative_samples=1,
        seed=7,
    ).fit(_train(), _items())
    second = MatrixFactorizationRecommender(
        factors=4,
        epochs=3,
        negative_samples=1,
        seed=7,
    ).fit(_train(), _items())

    assert first.recommend("user_1", top_k=2) == second.recommend("user_1", top_k=2)


def test_user_history_embedding_returns_recommendations() -> None:
    recommender = UserHistoryEmbeddingRecommender().fit(_train(), _items())

    results = recommender.recommend("user_1", top_k=2)

    assert len(results) == 2


def test_user_history_embedding_excludes_seen_items() -> None:
    recommender = UserHistoryEmbeddingRecommender().fit(_train(), _items())

    results = recommender.recommend("user_1", top_k=4, exclude_seen=True)
    item_ids = [result["item_id"] for result in results]

    assert "item_001" not in item_ids
    assert "item_004" not in item_ids


def test_batch_recommend_returns_multiple_users() -> None:
    recommender = PopularityRecommender().fit(_train(), _items())

    results = recommender.batch_recommend(["user_1", "user_2"], top_k=2)

    assert set(results) == {"user_1", "user_2"}
    assert all(len(user_results) == 2 for user_results in results.values())


def test_unknown_user_does_not_crash() -> None:
    recommender = UserHistoryEmbeddingRecommender().fit(_train(), _items())

    results = recommender.recommend("unknown", top_k=2)

    assert len(results) == 2
