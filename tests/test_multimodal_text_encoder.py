from __future__ import annotations

import numpy as np
import pandas as pd

from src.multimodal.text_encoder import TextItemEncoder


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
                "description": "Cushioned stretching mat",
            },
            {
                "item_id": "i3",
                "title": "",
                "category": "",
                "brand": "",
                "description": "",
            },
        ]
    )


def test_text_encoder_shape_query_and_normalization() -> None:
    encoder = TextItemEncoder(max_features=20, normalize=True)
    vectors = encoder.fit_transform(_items())
    queries = encoder.encode_queries(["wireless audio"])
    assert vectors.shape[0] == 3
    assert queries.shape[1] == vectors.shape[1]
    norms = np.linalg.norm(vectors[:2], axis=1)
    np.testing.assert_allclose(norms, np.ones(2))


def test_text_encoder_deterministic_vocabulary_and_roundtrip(tmp_path) -> None:
    encoder = TextItemEncoder(max_features=20).fit(_items())
    assert encoder.get_feature_names() == sorted(encoder.get_feature_names())
    path = tmp_path / "text.pkl"
    encoder.save(path)
    loaded = TextItemEncoder.load(path)
    np.testing.assert_allclose(encoder.transform(_items()), loaded.transform(_items()))
