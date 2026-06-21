"""LLM-backed structured query understanding."""

from __future__ import annotations

from typing import Any

from src.llm.clients.base_client import BaseLLMClient
from src.llm.prompts.query_understanding_prompts import (
    build_query_understanding_prompt,
    build_query_understanding_system_prompt,
)
from src.llm.query_understanding.query_rewriter import (
    fallback_query_expansion,
    fallback_query_rewrite,
    normalize_expanded_queries,
)
from src.llm.query_understanding.schemas import validate_query_understanding_result
from src.llm.utils import safe_json_loads
from src.query_understanding.query_parser import parse_query


class LLMIntentExtractor:
    """Parse query structure using an LLM client with deterministic fallback."""

    def __init__(
        self,
        client: BaseLLMClient,
        known_categories: list[str] | None = None,
        known_brands: list[str] | None = None,
        max_expanded_queries: int = 3,
    ) -> None:
        self.client = client
        self.known_categories = known_categories or []
        self.known_brands = known_brands or []
        self.max_expanded_queries = int(max_expanded_queries)

    def _fallback(self, query_text: str) -> dict[str, Any]:
        parsed = parse_query(query_text, self.known_categories, self.known_brands)
        return {
            "intent": parsed.get("intent"),
            "category": parsed.get("category"),
            "brand": parsed.get("brand"),
            "price_constraint": parsed.get("price_constraint"),
            "use_case": parsed.get("use_case"),
            "rewritten_query": fallback_query_rewrite(query_text, parsed),
            "expanded_queries": fallback_query_expansion(
                query_text,
                parsed,
                max_queries=self.max_expanded_queries,
            ),
            "confidence": 0.0,
        }

    def parse_query(self, query_text: str) -> dict[str, Any]:
        """Parse one query and include LLM metadata."""
        prompt = build_query_understanding_prompt(
            query_text,
            known_categories=self.known_categories,
            known_brands=self.known_brands,
            max_expanded_queries=self.max_expanded_queries,
        )
        response = self.client.generate(
            prompt=prompt,
            system_prompt=build_query_understanding_system_prompt(),
            temperature=0.0,
            max_tokens=512,
            response_format="json",
        )
        raw_obj, parse_success, parse_error = safe_json_loads(response.text)
        schema_valid = False
        try:
            normalized = validate_query_understanding_result(raw_obj) if parse_success else {}
            schema_valid = parse_success
        except ValueError as exc:
            parse_error = str(exc)
            normalized = {}

        if not schema_valid:
            normalized = self._fallback(query_text)
        normalized["rewritten_query"] = normalized.get("rewritten_query") or fallback_query_rewrite(
            query_text,
            normalized,
        )
        normalized["expanded_queries"] = normalize_expanded_queries(
            normalized.get("expanded_queries", []),
            query_text,
            max_queries=self.max_expanded_queries,
        )
        normalized.update(
            {
                "parse_success": bool(parse_success),
                "schema_valid": bool(schema_valid),
                "parse_error": parse_error,
                "provider": response.provider,
                "model": response.model,
                "latency_ms": float(response.latency_ms),
                "estimated_cost_usd": float(response.estimated_cost_usd or 0.0),
            }
        )
        return normalized

    def batch_parse_queries(self, query_texts: list[str]) -> list[dict[str, Any]]:
        """Parse a batch of query strings."""
        return [self.parse_query(query_text) for query_text in query_texts]
