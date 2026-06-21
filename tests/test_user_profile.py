from __future__ import annotations

import numpy as np
import pandas as pd

from src.llm.clients.mock_client import MockLLMClient
from src.llm.user_modeling.profile_embedder import ProfileEmbedder
from src.llm.user_modeling.profile_generator import (
    LLMUserProfileGenerator,
    build_user_behavior_summary,
)
from src.llm.user_modeling.profile_pipeline import run_user_profile_generation


def _items() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item_id": "i1",
                "title": "Wireless Earbuds",
                "category": "Electronics",
                "brand": "Aster",
                "price": 99.0,
            },
            {
                "item_id": "i2",
                "title": "Yoga Mat",
                "category": "Sports",
                "brand": "Pulse",
                "price": 30.0,
            },
        ]
    )


def _train() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "click",
                "event_weight": 2.0,
                "timestamp": pd.Timestamp("2024-01-01"),
            },
            {
                "user_id": "u1",
                "item_id": "i2",
                "event_type": "purchase",
                "event_weight": 4.0,
                "timestamp": pd.Timestamp("2024-01-02"),
            },
        ]
    )


def test_behavior_summary_uses_actual_metadata() -> None:
    summary = build_user_behavior_summary("u1", _train(), _items())
    assert summary["top_categories"][0] == "Sports"
    assert "Pulse" in summary["top_brands"]
    assert summary["purchase_count"] == 1


def test_profile_generator_returns_valid_non_sensitive_profile() -> None:
    summary = build_user_behavior_summary("u1", _train(), _items())
    profile = LLMUserProfileGenerator(MockLLMClient()).generate_profile(summary)
    assert profile["schema_valid"]
    assert profile["profile_text"]
    assert "sensitive" not in profile["profile_text"].lower()


def test_profile_embedder_deterministic_and_unseen_safe() -> None:
    texts = ["interests electronics sports", "interests beauty"]
    embedder = ProfileEmbedder(max_features=10)
    first = embedder.fit_transform(texts)
    second = ProfileEmbedder(max_features=10).fit_transform(texts)
    np.testing.assert_allclose(first, second)
    unseen = embedder.transform(["completely unseen token"])
    assert unseen.shape == (1, first.shape[1])


def test_user_profile_pipeline_columns() -> None:
    config = {
        "seed": 42,
        "llm": {"client": "mock", "model": "mock-user-profile-v1"},
        "user_profile": {"min_history_items": 1, "max_recent_items": 2},
        "profile_embedding": {"method": "bag_of_words", "max_features": 16, "normalize": True},
    }
    profiles, embeddings = run_user_profile_generation(_train(), _items(), config)
    assert {"user_id", "profile_text", "parse_success", "history_length"}.issubset(profiles.columns)
    assert {"user_id", "embedding", "embedding_dim"}.issubset(embeddings.columns)
    assert embeddings["embedding_dim"].iloc[0] > 0
