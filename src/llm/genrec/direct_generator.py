"""Direct free-form GenRec baseline."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.llm.clients.base_client import BaseLLMClient
from src.llm.genrec.output_parser import parse_recommendation_output
from src.llm.prompts.genrec_prompts import (
    build_direct_generation_prompt,
    build_genrec_system_prompt,
)


class DirectGenerator:
    """Unsafe direct-generation baseline used only for validation comparison."""

    method = "direct_generation"

    def __init__(
        self,
        client: BaseLLMClient,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.client = client
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    def generate(
        self,
        query_text: str,
        parsed_query: dict | None = None,
        user_profile: dict | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Generate direct recommendations without candidate grounding."""
        prompt = build_direct_generation_prompt(
            query_text=query_text,
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
        parsed = parse_recommendation_output(response.text)
        parsed.update(
            {
                "method": self.method,
                "provider": response.provider,
                "model": response.model,
                "latency_ms": float(response.latency_ms),
                "estimated_cost_usd": float(response.estimated_cost_usd or 0.0),
            }
        )
        return parsed

    def batch_generate(
        self,
        contexts: list[dict[str, Any]],
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Generate one direct output per query context."""
        rows = []
        for context in contexts:
            output = self.generate(
                query_text=str(context.get("query_text", "")),
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
                    "candidate_pool_size": 0,
                }
            )
            rows.append(output)
        return pd.DataFrame(rows)
