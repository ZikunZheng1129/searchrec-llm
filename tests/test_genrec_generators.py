from __future__ import annotations

from src.llm.clients.base_client import BaseLLMClient, LLMResponse
from src.llm.clients.mock_client import MockLLMClient
from src.llm.genrec.constrained_generator import CandidateConstrainedGenerator
from src.llm.genrec.direct_generator import DirectGenerator


class InvalidItemClient(BaseLLMClient):
    provider = "test"
    model = "invalid"

    def generate(self, *args, **kwargs) -> LLMResponse:
        del args, kwargs
        return LLMResponse(
            text='{"recommended_item_ids": ["not_a_candidate"]}',
            provider=self.provider,
            model=self.model,
            latency_ms=0.1,
            estimated_cost_usd=0.0,
        )


def _contexts():
    return [
        {
            "query_id": "q1",
            "query_text": "beauty serum",
            "split": "test",
            "target_item_id": "item_00001",
            "candidates": [{"item_id": "item_00001"}, {"item_id": "item_00002"}],
        }
    ]


def test_direct_generator_returns_method_and_invalid_mock_id():
    generator = DirectGenerator(MockLLMClient(model="mock-genrec-v1"))
    output = generator.generate("beauty serum", top_k=3)
    assert output["method"] == "direct_generation"
    assert "item_mock_invalid_001" in output["recommended_item_ids"]


def test_candidate_constrained_generator_outputs_only_candidate_ids():
    generator = CandidateConstrainedGenerator(MockLLMClient(model="mock-genrec-v1"))
    output = generator.generate(
        "beauty serum",
        candidates=[{"item_id": "item_00001"}, {"item_id": "item_00002"}],
        top_k=2,
    )
    assert output["method"] == "candidate_constrained_generation"
    assert set(output["recommended_item_ids"]).issubset({"item_00001", "item_00002"})
    assert output["used_fallback"] is False


def test_candidate_constrained_generator_falls_back_on_invalid_output():
    generator = CandidateConstrainedGenerator(InvalidItemClient())
    output = generator.generate(
        "beauty serum",
        candidates=[{"item_id": "item_00001"}, {"item_id": "item_00002"}],
        top_k=2,
    )
    assert output["used_fallback"] is True
    assert output["invalid_item_ids"] == ["not_a_candidate"]
    assert output["recommended_item_ids"] == ["item_00001", "item_00002"]


def test_batch_generation_returns_one_row_per_query_and_no_external_calls():
    generator = CandidateConstrainedGenerator(MockLLMClient(model="mock-genrec-v1"))
    output = generator.batch_generate(_contexts(), top_k=2)
    assert len(output) == 1
    assert output["provider"].iloc[0] == "mock"
    assert output["estimated_cost_usd"].iloc[0] == 0.0
