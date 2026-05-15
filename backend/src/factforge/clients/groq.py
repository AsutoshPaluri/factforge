"""Groq LLM client — async, text-only.

Groq serves open-source models (Llama 3.3 70B) at ~200-500 tokens/s with
a generous free tier (~30 RPM, ~14k requests/day). We use it for:
  - text-only claim decomposition
  - summary generation

Gemini stays the primary client for vision-mode (image) inputs since
Groq doesn't expose multimodal endpoints yet.

API surface mirrors clients/gemini.py:
  - generate_text(prompt) -> str
  - generate_structured(prompt, schema) -> schema instance
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TypeVar

import structlog
from groq import AsyncGroq
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from factforge.config import settings

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class GroqUsage:
    """Token counts from one Groq call."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class GroqClient:
    """Async Groq client with retries and Pydantic structured output."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
        max_tokens: int = 1500,
        temperature: float = 0.3,
    ) -> None:
        self.api_key = api_key or settings.groq_api_key
        self.model = model or settings.groq_model
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.temperature = temperature

        if not self.api_key:
            raise ValueError(
                "Groq API key missing. Set GROQ_API_KEY in .env or pass api_key=..."
            )

        self._client = AsyncGroq(api_key=self.api_key)
        self._last_usage: GroqUsage | None = None

    @property
    def last_usage(self) -> GroqUsage | None:
        return self._last_usage

    def _record_usage(self, response: object) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            self._last_usage = None
            return
        self._last_usage = GroqUsage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
        )

    async def _create(
        self,
        messages: list[dict],
        response_format: dict | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ):
        """Single chat completion call with retry."""
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt:
                kwargs: dict = {
                    "model": model or self.model,
                    "messages": messages,
                    "temperature": (
                        temperature if temperature is not None else self.temperature
                    ),
                    "max_tokens": self.max_tokens,
                }
                if response_format:
                    kwargs["response_format"] = response_format
                response = await self._client.chat.completions.create(**kwargs)
                self._record_usage(response)
                return response
        raise RuntimeError("retry loop exited without return")

    async def generate_text(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Plain text in -> plain text out. Optional model + temperature overrides."""
        response = await self._create(
            [{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,
        )
        return (response.choices[0].message.content or "").strip()

    async def generate_structured(self, prompt: str, schema: type[T]) -> T:
        """Generate JSON conforming to a Pydantic schema and parse it.

        Groq supports JSON mode (`response_format={"type": "json_object"}`)
        which guarantees valid JSON. We add the schema description to the
        system message so the model knows what shape to produce.
        """
        schema_json = schema.model_json_schema()
        system_msg = (
            "You must respond with a single JSON object that matches this "
            "exact JSON Schema. Do not wrap it in markdown code fences. "
            "Do not include any text outside the JSON object.\n\n"
            f"JSON Schema:\n{json.dumps(schema_json)}"
        )

        response = await self._create(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = (response.choices[0].message.content or "").strip()

        # Strip optional ```json fences just in case
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        return schema.model_validate_json(content)


# --- Singleton accessor ---
_instance: GroqClient | None = None


def get_groq() -> GroqClient:
    """Process-wide GroqClient singleton (created on first call)."""
    global _instance
    if _instance is None:
        _instance = GroqClient()
    return _instance
