from __future__ import annotations

import pytest

from src.llm.clients.base_client import BaseLLMClient
from src.llm.clients.local_hf_client import LocalHFClient
from src.llm.clients.mock_client import MockLLMClient
from src.llm.clients.openai_client import OpenAIClient
from src.llm.prompts.query_understanding_prompts import build_query_understanding_prompt
from src.llm.prompts.user_profile_prompts import build_user_profile_prompt
from src.llm.utils import safe_json_loads


def test_base_client_generate_raises() -> None:
    with pytest.raises(NotImplementedError):
        BaseLLMClient().generate("hello")


def test_mock_client_returns_valid_query_json() -> None:
    prompt = build_query_understanding_prompt(
        "budget electronics for gym",
        known_categories=["Electronics"],
        known_brands=["Aster"],
    )
    response = MockLLMClient().generate(prompt)
    parsed, success, error = safe_json_loads(response.text)
    assert success, error
    assert parsed["intent"] == "price_sensitive_search"
    assert "expanded_queries" in parsed


def test_mock_client_returns_valid_profile_json() -> None:
    prompt = build_user_profile_prompt(
        "user_1",
        {
            "user_id": "user_1",
            "top_categories": ["Electronics"],
            "top_brands": ["Aster"],
            "behavior_signals": ["click"],
            "recent_interests": ["Wireless Earbuds"],
            "price_preference": "mid_range",
        },
    )
    response = MockLLMClient(model="mock-user-profile-v1").generate(prompt)
    parsed, success, error = safe_json_loads(response.text)
    assert success, error
    assert parsed["user_id"] == "user_1"
    assert parsed["profile_text"]


def test_mock_client_is_deterministic() -> None:
    prompt = build_query_understanding_prompt("pulse sports product")
    first = MockLLMClient(seed=7).generate(prompt).text
    second = MockLLMClient(seed=7).generate(prompt).text
    assert first == second


def test_safe_json_loads_variants() -> None:
    assert safe_json_loads('{"a": 1}')[1]
    assert safe_json_loads('```json\n{"a": 1}\n```')[1]
    parsed, success, error = safe_json_loads("{bad")
    assert parsed == {}
    assert not success
    assert error


def test_optional_clients_do_not_import_or_call_by_default() -> None:
    with pytest.raises(RuntimeError):
        OpenAIClient(model="gpt-test", allow_external_api_calls=False).generate("hello")
    with pytest.raises(RuntimeError):
        LocalHFClient(model="tiny", allow_model_load=False).generate("hello")
