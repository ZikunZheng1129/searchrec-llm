"""Template and LLM explanation generators."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.llm.clients.base_client import BaseLLMClient
from src.llm.explanations.evidence_selector import select_evidence_for_recommendations
from src.llm.explanations.hallucination_checker import check_explanation_faithfulness
from src.llm.genrec.output_parser import parse_explanation_output
from src.llm.prompts.genrec_prompts import (
    build_explanation_prompt,
    build_explanation_system_prompt,
)


def _template_explanations(query_id: str, evidence: list[dict]) -> dict[str, Any]:
    explanations = []
    for item in evidence:
        item_id = str(item.get("item_id", ""))
        fields = ["title", "category", "brand", "avg_rating"]
        explanations.append(
            {
                "item_id": item_id,
                "explanation": (
                    f"{item.get('title', item_id)} matches because it is a "
                    f"{item.get('category', 'catalog')} item from "
                    f"{item.get('brand', 'the catalog')} with rating "
                    f"{float(item.get('avg_rating', 0.0) or 0.0):.2f}."
                ),
                "evidence_item_ids": [item_id],
                "evidence_fields": fields,
                "confidence": 0.75,
            }
        )
    return {"query_id": query_id, "explanations": explanations}


class TemplateExplanationGenerator:
    """Deterministic non-LLM explanation generator."""

    method = "template_explanation"

    def generate(
        self,
        query_id: str,
        query_text: str,
        recommended_item_ids: list[str],
        items: pd.DataFrame,
        ranking_candidates_for_query: pd.DataFrame | None = None,
        parsed_query: dict | None = None,
        user_profile: dict | None = None,
        max_evidence_items: int = 5,
    ) -> dict[str, Any]:
        """Generate template explanations from selected evidence only."""
        del query_text, parsed_query, user_profile
        evidence = select_evidence_for_recommendations(
            recommended_item_ids,
            items,
            ranking_candidates_for_query,
            max_items=max_evidence_items,
        )
        parsed = _template_explanations(query_id, evidence)
        faithfulness = check_explanation_faithfulness(
            parsed["explanations"],
            {str(row["item_id"]): row for row in evidence},
        )
        parsed.update(
            {
                "method": self.method,
                "provider": "template",
                "model": "template",
                "parse_success": True,
                "schema_valid": True,
                "used_fallback": False,
                "parse_error": None,
                "latency_ms": 0.0,
                "estimated_cost_usd": 0.0,
                **faithfulness,
            }
        )
        return parsed


class LLMExplanationGenerator:
    """Evidence-grounded LLM explanation generator with template fallback."""

    method = "evidence_grounded_llm_explanation"

    def __init__(
        self,
        client: BaseLLMClient,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.client = client
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.template = TemplateExplanationGenerator()

    def generate(
        self,
        query_id: str,
        query_text: str,
        recommended_item_ids: list[str],
        items: pd.DataFrame,
        ranking_candidates_for_query: pd.DataFrame | None = None,
        parsed_query: dict | None = None,
        user_profile: dict | None = None,
        max_evidence_items: int = 5,
    ) -> dict[str, Any]:
        """Generate evidence-grounded explanations and fallback on invalid output."""
        evidence = select_evidence_for_recommendations(
            recommended_item_ids,
            items,
            ranking_candidates_for_query,
            max_items=max_evidence_items,
        )
        recommended_items = [
            {"item_id": str(item_id)} for item_id in recommended_item_ids[: int(max_evidence_items)]
        ]
        prompt = build_explanation_prompt(
            query_text=query_text,
            recommended_items=recommended_items,
            evidence=evidence,
            parsed_query=parsed_query,
            user_profile=user_profile,
        )
        response = self.client.generate(
            prompt=prompt,
            system_prompt=build_explanation_system_prompt(),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format="json",
        )
        parsed = parse_explanation_output(response.text)
        parsed["query_id"] = query_id
        evidence_lookup = {str(row["item_id"]): row for row in evidence}
        faithfulness = check_explanation_faithfulness(
            parsed.get("explanations", []),
            evidence_lookup,
        )
        needs_fallback = (
            not parsed.get("parse_success", False)
            or not parsed.get("schema_valid", False)
            or not faithfulness["explanation_faithful"]
        )
        if needs_fallback:
            fallback = self.template.generate(
                query_id=query_id,
                query_text=query_text,
                recommended_item_ids=recommended_item_ids,
                items=items,
                ranking_candidates_for_query=ranking_candidates_for_query,
                max_evidence_items=max_evidence_items,
            )
            fallback.update(
                {
                    "method": self.method,
                    "provider": response.provider,
                    "model": response.model,
                    "parse_success": bool(parsed.get("parse_success", False)),
                    "schema_valid": bool(parsed.get("schema_valid", False)),
                    "used_fallback": True,
                    "parse_error": parsed.get("parse_error"),
                    "latency_ms": float(response.latency_ms),
                    "estimated_cost_usd": float(response.estimated_cost_usd or 0.0),
                }
            )
            return fallback
        parsed.update(
            {
                "method": self.method,
                "provider": response.provider,
                "model": response.model,
                "used_fallback": False,
                "latency_ms": float(response.latency_ms),
                "estimated_cost_usd": float(response.estimated_cost_usd or 0.0),
                **faithfulness,
            }
        )
        return parsed
