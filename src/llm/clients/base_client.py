"""Provider-neutral LLM client interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    """Text response and lightweight metadata from an LLM call."""

    text: str
    provider: str
    model: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    estimated_cost_usd: float | None = None
    raw_response: dict[str, Any] | None = None


class BaseLLMClient:
    """Minimal dependency-free interface for text generation clients."""

    provider = "base"
    model = "base"

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: str | None = "json",
    ) -> LLMResponse:
        """Generate text for a prompt."""
        raise NotImplementedError("BaseLLMClient.generate must be implemented by subclasses")
