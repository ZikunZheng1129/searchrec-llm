import numpy as np

from src.query_understanding.intent_classifier import classify_intent
from src.query_understanding.query_encoder import BagOfWordsQueryEncoder
from src.query_understanding.query_parser import normalize_query, parse_query


def test_normalize_query_lowercases_and_collapses_whitespace() -> None:
    assert normalize_query("  Wireless   Earbuds  ") == "wireless earbuds"


def test_classify_intent_detects_price_sensitive_queries() -> None:
    assert classify_intent("cheap electronics under 50") == "price_sensitive_search"


def test_classify_intent_detects_use_case_queries() -> None:
    assert classify_intent("running shoes for beginners") == "use_case_search"


def test_parse_query_extracts_category_from_known_categories() -> None:
    parsed = parse_query(
        "best electronics for travel",
        known_categories=["Sports & Outdoors", "Electronics"],
    )

    assert parsed["category"] == "Electronics"


def test_parse_query_extracts_brand_from_known_brands() -> None:
    parsed = parse_query("Aster beauty product", known_brands=["Aster", "Nova"])

    assert parsed["brand"] == "Aster"


def test_parse_query_extracts_price_constraint() -> None:
    parsed = parse_query("premium electronics under 50")

    assert parsed["price_constraint"] == "under_50"


def test_parse_query_extracts_use_case() -> None:
    parsed = parse_query("budget skincare for dry skin")

    assert parsed["use_case"] == "dry skin"


def test_bag_of_words_query_encoder_fit_transform_shape() -> None:
    encoder = BagOfWordsQueryEncoder()
    matrix = encoder.fit_transform(["wireless earbuds", "wireless charger"])

    assert matrix.shape == (2, 3)
    assert encoder.get_feature_names() == ["charger", "earbuds", "wireless"]


def test_bag_of_words_query_encoder_transform_handles_unseen_tokens() -> None:
    encoder = BagOfWordsQueryEncoder().fit(["wireless earbuds"])
    matrix = encoder.transform(["wireless unknown token"])

    assert matrix.shape == (1, 2)
    assert np.array_equal(matrix, np.array([[0.0, 1.0]]))
