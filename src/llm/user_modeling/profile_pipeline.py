"""Pipeline helpers for Stage 8 user profile generation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.llm.query_understanding.query_understanding_pipeline import build_llm_client_from_config
from src.llm.user_modeling.profile_embedder import ProfileEmbedder
from src.llm.user_modeling.profile_generator import (
    LLMUserProfileGenerator,
    build_all_user_behavior_summaries,
)


def run_user_profile_generation(
    train_interactions: pd.DataFrame,
    items: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate user profiles and lightweight profile embeddings."""
    profile_config = config.get("user_profile", {})
    summaries = build_all_user_behavior_summaries(
        train_interactions=train_interactions,
        items=items,
        min_history_items=int(profile_config.get("min_history_items", 1)),
        max_recent_items=int(profile_config.get("max_recent_items", 5)),
    )
    max_users = profile_config.get("max_users")
    if max_users is not None:
        summaries = summaries[: int(max_users)]

    client = build_llm_client_from_config(config)
    generator = LLMUserProfileGenerator(
        client,
        max_recent_items=int(profile_config.get("max_recent_items", 5)),
    )
    profiles = generator.batch_generate_profiles(summaries)

    embedding_config = config.get("profile_embedding", {})
    embedder = ProfileEmbedder(
        max_features=int(embedding_config.get("max_features", 512)),
        normalize=bool(embedding_config.get("normalize", True)),
    )
    profile_texts = profiles["profile_text"].fillna("").astype(str).tolist()
    vectors = embedder.fit_transform(profile_texts)
    embeddings = pd.DataFrame(
        {
            "user_id": profiles["user_id"].astype(str),
            "profile_text": profile_texts,
            "embedding": [row.astype(float).tolist() for row in vectors],
            "embedding_dim": int(vectors.shape[1]),
            "embedding_method": str(embedding_config.get("method", "bag_of_words")),
        }
    )
    return profiles, embeddings
