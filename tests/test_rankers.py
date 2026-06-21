from __future__ import annotations

import numpy as np
import pandas as pd

from src.ranking.cross_encoder_ranker import CrossEncoderRankerWrapper
from src.ranking.dataset import get_numeric_feature_columns
from src.ranking.feature_ranker import NumpyLinearRanker
from src.ranking.lightgbm_ranker import LightGBMRanker
from src.ranking.mixed_ranker import MixedRanker
from src.ranking.mlp_ranker import MLPRankerWrapper


def _ranking_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_id": "q1",
                "query_text": "wireless earbuds",
                "split": "train",
                "target_item_id": "i1",
                "candidate_item_id": "i1",
                "label": 1,
                "hybrid_score": 0.9,
                "authority_score": 0.8,
                "conversion_proxy": 0.4,
                "cold_start_score": 0.1,
                "category": "Electronics",
            },
            {
                "query_id": "q1",
                "query_text": "wireless earbuds",
                "split": "train",
                "target_item_id": "i1",
                "candidate_item_id": "i2",
                "label": 0,
                "hybrid_score": 0.2,
                "authority_score": 0.3,
                "conversion_proxy": 0.0,
                "cold_start_score": 0.8,
                "category": "Beauty",
            },
            {
                "query_id": "q2",
                "query_text": "hydrating serum",
                "split": "train",
                "target_item_id": "i3",
                "candidate_item_id": "i3",
                "label": 1,
                "hybrid_score": 0.8,
                "authority_score": 0.7,
                "conversion_proxy": 0.2,
                "cold_start_score": 0.2,
                "category": "Beauty",
            },
            {
                "query_id": "q2",
                "query_text": "hydrating serum",
                "split": "train",
                "target_item_id": "i3",
                "candidate_item_id": "i2",
                "label": 0,
                "hybrid_score": 0.1,
                "authority_score": 0.3,
                "conversion_proxy": 0.0,
                "cold_start_score": 0.8,
                "category": "Beauty",
            },
        ]
    )


def _items() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item_id": "i1",
                "title": "Wireless Earbuds",
                "category": "Electronics",
                "brand": "Aster",
                "description": "Clear audio",
            },
            {
                "item_id": "i2",
                "title": "Yoga Mat",
                "category": "Sports",
                "brand": "Pulse",
                "description": "Stretching mat",
            },
            {
                "item_id": "i3",
                "title": "Hydrating Serum",
                "category": "Beauty",
                "brand": "Haven",
                "description": "Gentle skincare",
            },
        ]
    )


def test_numpy_linear_ranker_save_load(tmp_path) -> None:
    df = _ranking_df()
    features = get_numeric_feature_columns(df)
    ranker = NumpyLinearRanker(num_epochs=5).fit(df, features)
    scores = ranker.predict(df)
    assert np.isfinite(scores).all()
    path = tmp_path / "ranker.pkl"
    ranker.save(path)
    loaded = NumpyLinearRanker.load(path)
    np.testing.assert_allclose(scores, loaded.predict(df))


def test_lightgbm_ranker_numpy_fallback() -> None:
    df = _ranking_df()
    features = get_numeric_feature_columns(df)
    ranker = LightGBMRanker(backend="numpy_linear", num_epochs_fallback=5).fit(df, features)
    assert ranker.backend == "numpy_linear"
    assert np.isfinite(ranker.predict(df)).all()


def test_mlp_ranker_train_predict_save_load(tmp_path) -> None:
    df = _ranking_df()
    features = get_numeric_feature_columns(df)
    config = {
        "seed": 1,
        "model": {"hidden_dims": [8], "dropout": 0.0},
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 0.01, "device": "cpu"},
    }
    ranker = MLPRankerWrapper(config).fit(df, features)
    scores = ranker.predict(df)
    assert np.isfinite(scores).all()
    path = tmp_path / "mlp.pt"
    ranker.save(path)
    loaded = MLPRankerWrapper.load(path)
    assert np.isfinite(loaded.predict(df)).all()


def test_cross_encoder_ranker_train_predict_save_load(tmp_path) -> None:
    df = _ranking_df()
    config = {
        "seed": 1,
        "model": {
            "max_pair_len": 16,
            "vocab_size": 100,
            "embedding_dim": 8,
            "num_heads": 2,
            "num_layers": 1,
            "dropout": 0.0,
        },
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 0.01, "device": "cpu"},
    }
    ranker = CrossEncoderRankerWrapper(config).fit(df, _items())
    scores = ranker.predict(df, _items())
    assert np.isfinite(scores).all()
    path = tmp_path / "cross.pt"
    ranker.save(path)
    loaded = CrossEncoderRankerWrapper.load(path)
    assert np.isfinite(loaded.predict(df, _items())).all()


def test_mixed_ranker_is_deterministic_and_save_load(tmp_path) -> None:
    df = _ranking_df()
    ranker = MixedRanker().fit(df)
    first = ranker.predict(df)
    second = ranker.predict(df)
    np.testing.assert_allclose(first, second)
    path = tmp_path / "mixed.pkl"
    ranker.save(path)
    loaded = MixedRanker.load(path)
    np.testing.assert_allclose(first, loaded.predict(df))
