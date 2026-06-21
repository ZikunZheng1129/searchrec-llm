"""Deterministic mock LLM client for local tests and development."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from src.llm.clients.base_client import BaseLLMClient, LLMResponse
from src.llm.utils import estimate_mock_cost, estimate_token_count, safe_json_loads
from src.query_understanding.query_parser import parse_query


class MockLLMClient(BaseLLMClient):
    """A deterministic local client that returns valid structured JSON."""

    provider = "mock"

    def __init__(
        self,
        model: str = "mock-query-understanding-v1",
        seed: int = 42,
        simulate_invalid_json: bool = False,
    ) -> None:
        self.model = model
        self.seed = int(seed)
        self.simulate_invalid_json = bool(simulate_invalid_json)

    @staticmethod
    def _extract_line(prompt: str, label: str) -> str:
        match = re.search(rf"^{re.escape(label)}:\s*(.*)$", prompt, flags=re.MULTILINE)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_json_after(prompt: str, label: str) -> dict[str, Any]:
        match = re.search(rf"{re.escape(label)}:\s*(\{{.*\}})", prompt, flags=re.DOTALL)
        if not match:
            return {}
        parsed, success, _ = safe_json_loads(match.group(1))
        return parsed if success else {}

    @staticmethod
    def _extract_json_line(prompt: str, label: str) -> dict[str, Any]:
        line = MockLLMClient._extract_line(prompt, label)
        if not line:
            return {}
        parsed, success, _ = safe_json_loads(line)
        return parsed if success else {}

    @staticmethod
    def _split_known_values(prompt: str, label: str) -> list[str]:
        line = MockLLMClient._extract_line(prompt, label)
        if not line or line.lower() in {"none", "null", "[]"}:
            return []
        return [value.strip() for value in line.split("|") if value.strip()]

    @staticmethod
    def _expanded_queries(
        query_text: str,
        category: str | None,
        brand: str | None,
        price: str | None,
        use_case: str | None,
        max_queries: int,
    ) -> list[str]:
        candidates = [
            query_text,
            " ".join(value for value in [price, category, use_case] if value),
            " ".join(value for value in [brand, category] if value),
            " ".join(value for value in [use_case, category] if value),
        ]
        normalized: list[str] = []
        for candidate in candidates:
            text = re.sub(r"\s+", " ", str(candidate).strip().lower())
            if text and text not in normalized:
                normalized.append(text)
        return normalized[:max_queries]

    def _query_understanding_json(self, prompt: str) -> dict[str, Any]:
        query_text = self._extract_line(prompt, "Query")
        max_expanded = int(self._extract_line(prompt, "Max expanded queries") or 3)
        known_categories = self._split_known_values(prompt, "Known categories")
        known_brands = self._split_known_values(prompt, "Known brands")
        parsed = parse_query(query_text, known_categories, known_brands)
        category = parsed.get("category")
        brand = parsed.get("brand")
        price = parsed.get("price_constraint")
        use_case = parsed.get("use_case")
        rewritten = " ".join(
            value for value in [price, brand, category, use_case] if value
        ) or parsed.get("normalized_query", query_text.lower())
        rewritten = re.sub(r"\s+", " ", rewritten.strip().lower())
        return {
            "intent": parsed.get("intent") or "unknown",
            "category": category,
            "brand": brand,
            "constraints": {"price": price, "use_case": use_case},
            "rewritten_query": rewritten,
            "expanded_queries": self._expanded_queries(
                query_text,
                category,
                brand,
                price,
                use_case,
                max_expanded,
            ),
            "confidence": 0.9,
        }

    @staticmethod
    def _price_preference(avg_price: float | None) -> str:
        if avg_price is None:
            return "unknown"
        if avg_price < 50:
            return "budget"
        if avg_price < 150:
            return "mid_range"
        return "premium"

    def _user_profile_json(self, prompt: str) -> dict[str, Any]:
        summary = self._extract_json_after(prompt, "Behavior summary")
        user_id = str(summary.get("user_id", "unknown"))
        top_categories = [str(value) for value in summary.get("top_categories", [])]
        top_brands = [str(value) for value in summary.get("top_brands", [])]
        signals = [str(value) for value in summary.get("behavior_signals", [])]
        recent = [str(value) for value in summary.get("recent_interests", [])]
        price_preference = str(
            summary.get("price_preference") or self._price_preference(summary.get("avg_price"))
        )
        categories_text = ", ".join(top_categories[:2]) or "observed products"
        brands_text = ", ".join(top_brands[:2]) or "observed brands"
        profile_text = (
            f"Interests: {categories_text}. Brands: {brands_text}. "
            f"Price preference: {price_preference}."
        )
        return {
            "user_id": user_id,
            "summary": (
                f"User shows interest in {categories_text}, based only on observed "
                "synthetic product interactions."
            ),
            "top_categories": top_categories,
            "top_brands": top_brands,
            "price_preference": price_preference,
            "behavior_signals": signals,
            "recent_interests": recent,
            "profile_text": profile_text,
            "confidence": 0.85,
        }

    @staticmethod
    def _top_k(prompt: str) -> int:
        line = MockLLMClient._extract_line(prompt, "Top K")
        try:
            return max(1, int(line))
        except ValueError:
            return 10

    @staticmethod
    def _ranked_items(item_ids: list[str], reason: str) -> list[dict[str, Any]]:
        return [
            {
                "item_id": item_id,
                "rank": rank,
                "reason": reason,
                "confidence": 0.8,
            }
            for rank, item_id in enumerate(item_ids, start=1)
        ]

    def _candidate_ids(self, prompt: str) -> list[str]:
        payload = self._extract_json_line(prompt, "Candidates JSON")
        candidates = payload.get("candidates", []) if isinstance(payload, dict) else []
        if not isinstance(candidates, list):
            return []
        seen: set[str] = set()
        item_ids = []
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("item_id") is not None:
                item_id = str(candidate["item_id"])
                if item_id not in seen:
                    seen.add(item_id)
                    item_ids.append(item_id)
        return item_ids

    def _direct_generation_json(self, prompt: str) -> dict[str, Any]:
        top_k = self._top_k(prompt)
        item_ids = [f"item_{index:05d}" for index in range(1, top_k)]
        item_ids.insert(1, "item_mock_invalid_001")
        item_ids = item_ids[:top_k]
        return {
            "recommended_item_ids": item_ids,
            "ranked_items": self._ranked_items(
                item_ids,
                "Direct mock generation without catalog grounding.",
            ),
        }

    def _candidate_constrained_json(self, prompt: str) -> dict[str, Any]:
        top_k = self._top_k(prompt)
        item_ids = self._candidate_ids(prompt)[:top_k]
        return {
            "recommended_item_ids": item_ids,
            "ranked_items": self._ranked_items(
                item_ids,
                "Selected from provided candidate IDs.",
            ),
        }

    def _rerank_json(self, prompt: str) -> dict[str, Any]:
        top_k = self._top_k(prompt)
        item_ids = self._candidate_ids(prompt)
        reranked = sorted(item_ids, reverse=True)[:top_k]
        return {
            "recommended_item_ids": reranked,
            "ranked_items": self._ranked_items(
                reranked,
                "Deterministically reranked from provided candidate IDs.",
            ),
        }

    def _explanation_json(self, prompt: str) -> dict[str, Any]:
        recommended_payload = self._extract_json_line(prompt, "Recommended items JSON")
        evidence_payload = self._extract_json_line(prompt, "Evidence JSON")
        recommended = recommended_payload.get("recommended_items", [])
        evidence = evidence_payload.get("evidence", [])
        evidence_by_id = {str(row.get("item_id")): row for row in evidence if isinstance(row, dict)}
        explanations = []
        for item in recommended:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("item_id", ""))
            row = evidence_by_id.get(item_id, {})
            explanations.append(
                {
                    "item_id": item_id,
                    "explanation": (
                        f"{row.get('title', item_id)} matches the query using catalog evidence "
                        f"from {row.get('category', 'the catalog')} and "
                        f"{row.get('brand', 'brand')}."
                    ),
                    "evidence_item_ids": [item_id],
                    "evidence_fields": ["title", "category", "brand", "avg_rating"],
                    "confidence": 0.82,
                }
            )
        return {"query_id": "", "explanations": explanations}

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: str | None = "json",
    ) -> LLMResponse:
        """Return deterministic structured JSON for known Stage 8 prompts."""
        del temperature, max_tokens, response_format
        start = time.perf_counter()
        if self.simulate_invalid_json:
            text = "{invalid json"
        elif "GENREC_TASK: DIRECT_GENERATION" in prompt:
            text = json.dumps(self._direct_generation_json(prompt), sort_keys=True)
        elif "GENREC_TASK: CANDIDATE_CONSTRAINED_GENERATION" in prompt:
            text = json.dumps(self._candidate_constrained_json(prompt), sort_keys=True)
        elif "GENREC_TASK: LLM_RERANK" in prompt:
            text = json.dumps(self._rerank_json(prompt), sort_keys=True)
        elif "GENREC_TASK: EVIDENCE_GROUNDED_EXPLANATION" in prompt:
            text = json.dumps(self._explanation_json(prompt), sort_keys=True)
        elif "USER_PROFILE_TASK" in prompt or "USER_PROFILE_TASK" in str(system_prompt):
            text = json.dumps(self._user_profile_json(prompt), sort_keys=True)
        else:
            text = json.dumps(self._query_understanding_json(prompt), sort_keys=True)
        latency_ms = (time.perf_counter() - start) * 1000.0
        prompt_tokens = estimate_token_count(f"{system_prompt or ''} {prompt}")
        completion_tokens = estimate_token_count(text)
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimate_mock_cost(
                prompt_tokens,
                completion_tokens,
                self.provider,
                self.model,
            ),
            raw_response={"seed": self.seed},
        )
