from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.multimodal.image_encoder import (
    ImageItemEncoder,
    PrecomputedImageFeatureEncoder,
    TestStubImageEncoder,
)


def test_default_image_encoder_has_zero_availability() -> None:
    items = pd.DataFrame([{"item_id": "i1"}])
    encoder = ImageItemEncoder()
    assert encoder.transform(items) is None
    assert encoder.image_available_rate(items) == 0.0


def test_precomputed_image_encoder_reads_embeddings() -> None:
    items = pd.DataFrame(
        [
            {"item_id": "i1", "image_embedding": [1.0, 0.0]},
            {"item_id": "i2", "image_embedding": [0.0, 1.0]},
        ]
    )
    encoder = PrecomputedImageFeatureEncoder().fit(items)
    vectors = encoder.transform(items)
    assert vectors is not None
    assert vectors.shape == (2, 2)
    assert encoder.image_available_rate(items) == 1.0


def test_test_stub_image_encoder_is_explicit_only() -> None:
    with pytest.raises(ValueError):
        TestStubImageEncoder()
    items = pd.DataFrame([{"item_id": "i1"}, {"item_id": "i2"}])
    vectors = TestStubImageEncoder(allow_test_stub=True, dim=4).transform(items)
    assert vectors.shape == (2, 4)
    assert np.isfinite(vectors).all()
