from __future__ import annotations

from src.llm.genrec.output_parser import (
    fallback_recommendations_from_candidates,
    normalize_recommendation_items,
    parse_recommendation_output,
    validate_candidate_constrained_output,
)


def test_parse_plain_and_fenced_json_recommendation_output():
    plain = parse_recommendation_output('{"recommended_item_ids": ["a"], "ranked_items": []}')
    fenced = parse_recommendation_output(
        '```json\n{"recommended_item_ids": ["b"], "ranked_items": []}\n```'
    )
    assert plain["parse_success"] is True
    assert fenced["recommended_item_ids"] == ["b"]


def test_malformed_json_returns_parse_failure():
    parsed = parse_recommendation_output("{bad json")
    assert parsed["parse_success"] is False
    assert parsed["schema_valid"] is False
    assert parsed["parse_error"]


def test_duplicates_removed_and_invalid_candidate_ids_detected():
    parsed = normalize_recommendation_items(
        {"recommended_item_ids": ["a", "a", "x"], "parse_success": True},
        candidate_item_ids={"a", "b"},
        top_k=10,
    )
    assert parsed["recommended_item_ids"] == ["a"]
    assert parsed["invalid_item_ids"] == ["x"]
    assert parsed["schema_valid"] is False


def test_candidate_constrained_validation_and_fallback():
    parsed = parse_recommendation_output('{"recommended_item_ids": ["x", "a"]}')
    validated = validate_candidate_constrained_output(parsed, {"a"})
    assert validated["recommended_item_ids"] == ["a"]
    assert validated["invalid_item_ids"] == ["x"]
    fallback = fallback_recommendations_from_candidates(
        [{"item_id": "a"}, {"item_id": "b"}],
        top_k=1,
    )
    assert fallback["recommended_item_ids"] == ["a"]
    assert fallback["used_fallback"] is True
