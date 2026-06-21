from __future__ import annotations

import numpy as np
import pandas as pd

from src.multimodal.metadata_encoder import MetadataItemEncoder


def _items() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item_id": "i1",
                "category": "Electronics",
                "brand": "Aster",
                "price": 99.0,
                "avg_rating": 4.5,
                "rating_count": 100,
            },
            {
                "item_id": "i2",
                "category": "Sports",
                "brand": "Pulse",
                "price": None,
                "avg_rating": 4.0,
                "rating_count": 0,
            },
        ]
    )


def test_metadata_encoder_one_hot_numeric_and_unseen() -> None:
    encoder = MetadataItemEncoder(
        categorical_fields=["category", "brand"],
        numeric_fields=["price", "avg_rating", "rating_count"],
        add_log_numeric=["rating_count"],
    ).fit(_items())
    vectors = encoder.transform(_items())
    unseen = pd.DataFrame(
        [
            {
                "item_id": "i3",
                "category": "Unknown",
                "brand": "New",
                "price": np.nan,
                "avg_rating": np.nan,
                "rating_count": np.nan,
            }
        ]
    )
    unseen_vectors = encoder.transform(unseen)
    assert vectors.shape[1] == len(encoder.get_feature_names())
    assert np.isfinite(vectors).all()
    assert np.isfinite(unseen_vectors).all()
    assert "category=Electronics" in encoder.get_feature_names()


def test_metadata_encoder_roundtrip_and_deterministic_features(tmp_path) -> None:
    encoder = MetadataItemEncoder(["category"], ["price"]).fit(_items())
    assert encoder.get_feature_names() == sorted(encoder.get_feature_names())
    path = tmp_path / "metadata.pkl"
    encoder.save(path)
    loaded = MetadataItemEncoder.load(path)
    np.testing.assert_allclose(encoder.transform(_items()), loaded.transform(_items()))
