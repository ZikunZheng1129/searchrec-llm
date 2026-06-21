from __future__ import annotations

from src.llm.clients.base_client import BaseLLMClient, LLMResponse
from src.llm.clients.mock_client import MockLLMClient
from src.llm.genrec.llm_reranker import LLMReranker


class InvalidRerankClient(BaseLLMClient):
    provider = "test"
    model = "invalid-rerank"

    def generate(self, *args, **kwargs) -> LLMResponse:
        del args, kwargs
        return LLMResponse(
            text='{"recommended_item_ids": ["outside"]}',
            provider=self.provider,
            model=self.model,
            latency_ms=0.2,
            estimated_cost_usd=0.0,
        )


def _candidates(n: int = 20) -> list[dict]:
    return [{"item_id": f"item_{index:05d}"} for index in range(1, n + 1)]


def test_llm_reranker_reranks_and_outputs_candidate_ids():
    reranker = LLMReranker(MockLLMClient(model="mock-genrec-v1"), candidate_pool_size=10)
    output = reranker.rerank("query", _candidates(10), top_k=5)
    assert output["method"] == "llm_rerank_top_10"
    assert output["recommended_item_ids"][0] == "item_00010"
    assert set(output["recommended_item_ids"]).issubset(
        {candidate["item_id"] for candidate in _candidates(10)}
    )


def test_llm_reranker_invalid_output_falls_back_to_original_order():
    reranker = LLMReranker(InvalidRerankClient(), candidate_pool_size=10)
    output = reranker.rerank("query", _candidates(10), top_k=3)
    assert output["used_fallback"] is True
    assert output["recommended_item_ids"] == ["item_00001", "item_00002", "item_00003"]


def test_llm_reranker_top_20_batch_includes_metadata():
    reranker = LLMReranker(MockLLMClient(model="mock-genrec-v1"), candidate_pool_size=20)
    output = reranker.batch_rerank(
        [
            {
                "query_id": "q1",
                "query_text": "query",
                "split": "test",
                "target_item_id": "item_00001",
                "candidates": _candidates(20),
            }
        ],
        top_k=10,
    )
    assert output["method"].iloc[0] == "llm_rerank_top_20"
    assert bool(output["parse_success"].iloc[0]) is True
    assert output["provider"].iloc[0] == "mock"
    assert output["model"].iloc[0] == "mock-genrec-v1"
