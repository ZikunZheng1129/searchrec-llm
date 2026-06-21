"""Candidate-constrained LLM reranking."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.llm.clients.base_client import BaseLLMClient
from src.llm.genrec.output_parser import (
    fallback_recommendations_from_candidates,
    parse_rerank_output,
    validate_candidate_constrained_output,
)
from src.llm.prompts.genrec_prompts import build_genrec_system_prompt, build_llm_rerank_prompt


class LLMReranker:
    """Rerank top-N candidates while enforcing candidate item IDs."""

    def __init__(
        self,
        client: BaseLLMClient,
        candidate_pool_size: int = 10,
        method: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        allow_fallback: bool = True,
    ) -> None:
        self.client = client
        self.candidate_pool_size = int(candidate_pool_size)
        self.method = method or f"llm_rerank_top_{self.candidate_pool_size}"
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.allow_fallback = bool(allow_fallback)

    def rerank(
        self,
        query_text: str,
        candidates: list[dict],
        parsed_query: dict | None = None,
        user_profile: dict | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Rerank candidates and fallback to original order on invalid output."""
        prompt_candidates = candidates[: self.candidate_pool_size]
        candidate_item_ids = {str(candidate["item_id"]) for candidate in prompt_candidates}
        prompt = build_llm_rerank_prompt(
            query_text=query_text,
            candidates=prompt_candidates,
            parsed_query=parsed_query,
            user_profile=user_profile,
            top_k=top_k,
        )
        response = self.client.generate(
            prompt=prompt,
            system_prompt=build_genrec_system_prompt(),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format="json",
        )
        parsed = parse_rerank_output(response.text)
        validated = validate_candidate_constrained_output(parsed, candidate_item_ids)
        invalid_item_ids = list(validated.get("invalid_item_ids", []))
        needs_fallback = (
            not validated.get("parse_success", False)
            or not validated.get("schema_valid", False)
            or bool(invalid_item_ids)
        )
        if needs_fallback and self.allow_fallback:
            fallback = fallback_recommendations_from_candidates(prompt_candidates, top_k=top_k)
            fallback["invalid_item_ids"] = invalid_item_ids
            fallback["parse_success"] = bool(validated.get("parse_success", False))
            fallback["schema_valid"] = bool(validated.get("schema_valid", False))
            fallback["parse_error"] = validated.get("parse_error")
            validated = fallback
        validated.update(
            {
                "method": self.method,
                "provider": response.provider,
                "model": response.model,
                "latency_ms": float(response.latency_ms),
                "estimated_cost_usd": float(response.estimated_cost_usd or 0.0),
            }
        )
        return validated

    def batch_rerank(
        self,
        contexts: list[dict[str, Any]],
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Rerank candidate lists for query contexts."""
        rows = []
        for context in contexts:
            candidates = list(context.get("candidates", []))[: self.candidate_pool_size]
            output = self.rerank(
                query_text=str(context.get("query_text", "")),
                candidates=candidates,
                parsed_query=context.get("parsed_query"),
                user_profile=context.get("user_profile"),
                top_k=top_k,
            )
            output.update(
                {
                    "query_id": str(context.get("query_id", "")),
                    "query_text": str(context.get("query_text", "")),
                    "split": str(context.get("split", "")),
                    "target_item_id": str(context.get("target_item_id", "")),
                    "candidate_pool_size": len(candidates),
                }
            )
            rows.append(output)
        return pd.DataFrame(rows)
