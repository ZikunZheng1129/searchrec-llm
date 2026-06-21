import pandas as pd
import pytest

from src.evaluation.recommendation_eval import (
    coverage_at_k,
    evaluate_recommender,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    summarize_recommendation_metrics,
)
from src.recommendation.popularity import PopularityRecommender


def test_hit_rate_at_k() -> None:
    assert hit_rate_at_k(["item_a", "item_b"], {"item_b"}, 2) == 1.0
    assert hit_rate_at_k(["item_a"], {"item_b"}, 1) == 0.0


def test_recall_at_k() -> None:
    assert recall_at_k(["item_a", "item_b"], {"item_b", "item_c"}, 2) == 0.5


def test_ndcg_at_k() -> None:
    assert ndcg_at_k(["item_a", "item_b"], {"item_b"}, 2) == pytest.approx(1.0 / 1.5849625007211563)


def test_mrr_at_k() -> None:
    assert mrr_at_k(["item_a", "item_b"], {"item_b"}, 2) == 0.5


def test_coverage_at_k() -> None:
    recommendations = {"u1": ["a", "b"], "u2": ["b", "c"]}

    assert coverage_at_k(recommendations, {"a", "b", "c", "d"}, 2) == 0.75


def test_evaluate_recommender_returns_expected_columns() -> None:
    items = pd.DataFrame({"item_id": ["item_a", "item_b", "item_c"]})
    train = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "item_a", "event_weight": 3.0},
            {"user_id": "u2", "item_id": "item_b", "event_weight": 2.0},
        ]
    )
    test = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "item_c", "split": "test"},
            {"user_id": "u2", "item_id": "item_a", "split": "test"},
        ]
    )
    recommender = PopularityRecommender().fit(train, items)

    per_user = evaluate_recommender(
        recommender=recommender,
        eval_interactions=test,
        train_interactions=train,
        items=items,
        k_values=[1, 2],
        top_k=2,
    )

    assert {"hit_rate_at_1", "recall_at_2", "ndcg_at_2", "mrr_at_2"}.issubset(per_user.columns)
    summary = summarize_recommendation_metrics(per_user)
    assert summary.loc[0, "num_users"] == 2


def test_end_to_end_recommendation_evaluation_runs() -> None:
    items = pd.DataFrame(
        {
            "item_id": ["item_a", "item_b", "item_c"],
            "title": ["A", "B", "C"],
            "category": ["x", "x", "y"],
            "brand": ["b1", "b1", "b2"],
            "description": ["first", "second", "third"],
        }
    )
    train = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "item_a", "event_weight": 3.0},
            {"user_id": "u2", "item_id": "item_b", "event_weight": 2.0},
        ]
    )
    test = pd.DataFrame([{"user_id": "u1", "item_id": "item_c", "split": "test"}])
    recommender = PopularityRecommender().fit(train, items)

    per_user = evaluate_recommender(recommender, test, train, items, k_values=[2], top_k=2)
    summary = summarize_recommendation_metrics(per_user)

    assert not per_user.empty
    assert summary.loc[0, "split"] == "test"
