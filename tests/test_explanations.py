from __future__ import annotations

import pandas as pd

from src.llm.clients.base_client import BaseLLMClient, LLMResponse
from src.llm.clients.mock_client import MockLLMClient
from src.llm.explanations.evidence_selector import select_item_evidence
from src.llm.explanations.explanation_generator import (
    LLMExplanationGenerator,
    TemplateExplanationGenerator,
)
from src.llm.explanations.hallucination_checker import (
    check_candidate_constrained_item_ids,
    check_explanation_faithfulness,
    check_valid_item_ids,
)


class InvalidExplanationClient(BaseLLMClient):
    provider = "test"
    model = "bad-explanation"

    def generate(self, *args, **kwargs) -> LLMResponse:
        del args, kwargs
        return LLMResponse(
            text="{bad json",
            provider=self.provider,
            model=self.model,
            latency_ms=0.1,
            estimated_cost_usd=0.0,
        )


def _items() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["item_1"],
            "title": ["Serum"],
            "category": ["Beauty"],
            "brand": ["Haven"],
            "price": [12.0],
            "avg_rating": [4.5],
            "rating_count": [100],
            "description": ["Gentle serum"],
        }
    )


def test_evidence_selector_uses_item_metadata_and_candidate_fields():
    evidence = select_item_evidence("item_1", _items())
    assert evidence["title"] == "Serum"
    assert evidence["brand"] == "Haven"
    assert "Beauty" in evidence["evidence_text"]


def test_template_explanation_generator_returns_grounded_output():
    generator = TemplateExplanationGenerator()
    output = generator.generate("q1", "beauty", ["item_1"], _items())
    assert output["method"] == "template_explanation"
    assert output["explanations"][0]["item_id"] == "item_1"
    assert output["explanation_faithful"] is True


def test_llm_explanation_generator_uses_mock_client():
    generator = LLMExplanationGenerator(MockLLMClient(model="mock-genrec-v1"))
    output = generator.generate("q1", "beauty", ["item_1"], _items())
    assert output["method"] == "evidence_grounded_llm_explanation"
    assert output["provider"] == "mock"
    assert output["explanations"]


def test_llm_explanation_fallback_on_invalid_output():
    generator = LLMExplanationGenerator(InvalidExplanationClient())
    output = generator.generate("q1", "beauty", ["item_1"], _items())
    assert output["used_fallback"] is True
    assert output["explanations"][0]["item_id"] == "item_1"


def test_hallucination_checker_flags_invalid_ids_and_faithfulness():
    valid = check_valid_item_ids(["item_1", "bad"], {"item_1"})
    constrained = check_candidate_constrained_item_ids(["item_1", "bad"], {"item_1"})
    assert valid["invalid_item_ids"] == ["bad"]
    assert constrained["hallucination_rate"] == 0.5
    faithful = check_explanation_faithfulness(
        [
            {
                "item_id": "item_1",
                "explanation": "Beauty item from Haven",
                "evidence_item_ids": ["item_1"],
            }
        ],
        {"item_1": {"category": "Beauty", "brand": "Haven"}},
    )
    unfaithful = check_explanation_faithfulness(
        [{"item_id": "item_2", "explanation": "Other", "evidence_item_ids": ["item_2"]}],
        {"item_1": {"category": "Beauty", "brand": "Haven"}},
    )
    assert faithful["explanation_faithful"] is True
    assert unfaithful["explanation_faithful"] is False
