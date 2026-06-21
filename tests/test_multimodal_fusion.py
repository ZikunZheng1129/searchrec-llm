from __future__ import annotations

import numpy as np
import pandas as pd

from src.multimodal.fusion_model import MultimodalFusionEncoder
from src.multimodal.metadata_encoder import MetadataItemEncoder
from src.multimodal.multimodal_retriever import MultimodalRetriever
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
                "price": 99.0,
                "avg_rating": 4.5,
                "rating_count": 100,
            },
            {
                "item_id": "i2",
                "title": "Yoga Mat",
                "category": "Sports",
                "brand": "Pulse",
                "description": "Stretching mat",
                "price": 30.0,
                "avg_rating": 4.0,
                "rating_count": 0,
            },
        ]
    )


def _fusion() -> MultimodalFusionEncoder:
    return MultimodalFusionEncoder(
        text_encoder=TextItemEncoder(max_features=20),
        metadata_encoder=MetadataItemEncoder(["category", "brand"], ["price"]),
        normalize_output=True,
    )


def test_fusion_encoder_dimensions_and_normalization(tmp_path) -> None:
    encoder = _fusion()
    vectors = encoder.fit_transform(_items())
    query_vectors = encoder.encode_queries(["wireless audio"])
    assert vectors.shape[1] == query_vectors.shape[1]
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), np.ones(2))
    path = tmp_path / "fusion.pkl"
    encoder.save(path)
    loaded = MultimodalFusionEncoder.load(path)
    np.testing.assert_allclose(encoder.transform(_items()), loaded.transform(_items()))


def test_multimodal_retriever_deterministic_topk() -> None:
    items = _items()
    encoder = _fusion()
    vectors = encoder.fit_transform(items)
    retriever = MultimodalRetriever().fit(items, vectors, encoder)
    first = retriever.search("wireless audio", top_k=2)
    second = retriever.search("wireless audio", top_k=2)
    assert first == second
    assert first[0]["rank"] == 1
    assert {row["item_id"] for row in first} == {"i1", "i2"}
