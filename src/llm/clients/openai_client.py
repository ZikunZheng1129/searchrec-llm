"""Optional OpenAI client wrapper guarded from accidental local use."""

from __future__ import annotations

import os
import time

from src.llm.clients.base_client import BaseLLMClient, LLMResponse
from src.llm.utils import estimate_mock_cost, estimate_token_count


class OpenAIClient(BaseLLMClient):
    """Minimal lazy OpenAI wrapper.

    This client is optional and is never used by default configs or tests.
    """

    provider = "openai"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        allow_external_api_calls: bool = False,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.allow_external_api_calls = bool(allow_external_api_calls)

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: str | None = "json",
    ) -> LLMResponse:
        """Call OpenAI only when explicitly allowed."""
        if not self.allow_external_api_calls:
            raise RuntimeError("OpenAI calls are disabled; set allow_external_api_calls=true")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIClient")
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("Install the optional openai package to use OpenAIClient") from exc

        client = OpenAI(api_key=self.api_key)
        start = time.perf_counter()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0
        text = response.choices[0].message.content or ""
        prompt_tokens = getattr(getattr(response, "usage", None), "prompt_tokens", None)
        completion_tokens = getattr(getattr(response, "usage", None), "completion_tokens", None)
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens or estimate_token_count(prompt),
            completion_tokens=completion_tokens or estimate_token_count(text),
            estimated_cost_usd=estimate_mock_cost(
                prompt_tokens,
                completion_tokens,
                self.provider,
                self.model,
            ),
            raw_response={"id": getattr(response, "id", None)},
        )
