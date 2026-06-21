"""Optional local HuggingFace text-generation wrapper."""

from __future__ import annotations

import time

from src.llm.clients.base_client import BaseLLMClient, LLMResponse
from src.llm.utils import estimate_mock_cost, estimate_token_count


class LocalHFClient(BaseLLMClient):
    """Minimal lazy local-HF wrapper.

    The wrapper is optional and does not download or load models unless it is
    explicitly instantiated and used by a non-test config.
    """

    provider = "local_hf"

    def __init__(
        self,
        model: str,
        allow_model_load: bool = False,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.allow_model_load = bool(allow_model_load)
        self.device = device
        self._pipeline = None

    def _ensure_pipeline(self):
        if not self.allow_model_load:
            raise RuntimeError("Local HF model loading is disabled by default")
        if self._pipeline is None:
            try:
                from transformers import pipeline  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError(
                    "Install the optional transformers package to use LocalHFClient"
                ) from exc
            kwargs = {"model": self.model}
            if self.device is not None:
                kwargs["device"] = self.device
            self._pipeline = pipeline("text-generation", **kwargs)
        return self._pipeline

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: str | None = "json",
    ) -> LLMResponse:
        """Generate text with a locally loaded HF model when explicitly allowed."""
        del response_format
        generator = self._ensure_pipeline()
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        start = time.perf_counter()
        output = generator(
            full_prompt,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-6),
            num_return_sequences=1,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        text = str(output[0].get("generated_text", ""))
        prompt_tokens = estimate_token_count(full_prompt)
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
            raw_response={"local": True},
        )
