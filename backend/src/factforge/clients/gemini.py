"""Gemini client wrapper — text, structured-output, and multimodal.

Wraps `google.genai` with:
  - Async-first API (uses client.aio.*)
  - Pydantic-schema structured output
  - Multimodal input (text + image bytes)
  - Token usage tracking (for the daily-budget kill switch)
  - Retry on rate-limit / transient errors

Usage:
    client = GeminiClient()

    # Text in -> text out
    text = await client.generate_text("Say pong.")

    # Structured output
    class Sub(BaseModel):
        claims: list[str]
    out = await client.generate_structured("Decompose: <claim>", Sub)

    # Multimodal (image + text)
    text = await client.generate_multimodal(
        prompt="What's in this image?",
        image_bytes=b"...jpeg bytes...",
        image_mime="image/jpeg",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import structlog
from google import genai
from google.genai import types
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
class TokenUsage:
    """Token counts from one Gemini call. Pulled from response.usage_metadata."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class GeminiClient:
    """Thin async wrapper around google.genai with retries and token tracking."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError(
                "Gemini API key missing. Set GEMINI_API_KEY in .env or pass api_key=..."
            )

        self._client = genai.Client(api_key=self.api_key)
        self._last_usage: TokenUsage | None = None

    @property
    def last_usage(self) -> TokenUsage | None:
        """Token usage from the most recent successful call (for cost tracking)."""
        return self._last_usage

    # --- Internal helpers ---
    def _record_usage(self, response: object) -> None:
        """Pull token counts off the response and stash them."""
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            self._last_usage = None
            return
        self._last_usage = TokenUsage(
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            total_tokens=getattr(usage, "total_token_count", 0) or 0,
        )

    async def _generate(
        self,
        contents: list[object] | str,
        config: types.GenerateContentConfig | None = None,
    ) -> object:
        """Single generate call with retry."""
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt:
                response = await self._client.aio.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                self._record_usage(response)
                return response
        # Unreachable but mypy needs it
        raise RuntimeError("retry loop exited without return")

    # --- Public API ---
    async def generate_text(self, prompt: str) -> str:
        """Plain text in -> plain text out."""
        response = await self._generate(prompt)
        text = getattr(response, "text", None) or ""
        return text.strip()

    async def generate_structured(
        self,
        prompt: str,
        schema: type[T],
    ) -> T:
        """Generate JSON conforming to a Pydantic schema and parse it."""
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
        )
        response = await self._generate(prompt, config=config)

        # google.genai returns response.parsed when response_schema is set
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, schema):
            return parsed

        # Fallback: parse from text
        text = getattr(response, "text", None) or ""
        return schema.model_validate_json(text)

    async def generate_multimodal(
        self,
        prompt: str,
        image_bytes: bytes,
        image_mime: str = "image/jpeg",
        schema: type[T] | None = None,
    ) -> str | T:
        """Vision + text -> text (or structured if schema given)."""
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=image_mime)
        contents: list[object] = [image_part, prompt]

        config = None
        if schema is not None:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            )

        response = await self._generate(contents, config=config)

        if schema is not None:
            parsed = getattr(response, "parsed", None)
            if isinstance(parsed, schema):
                return parsed
            text = getattr(response, "text", None) or ""
            return schema.model_validate_json(text)

        text = getattr(response, "text", None) or ""
        return text.strip()


# --- Singleton accessor ---
_instance: GeminiClient | None = None


def get_gemini() -> GeminiClient:
    """Process-wide GeminiClient singleton (created on first call)."""
    global _instance
    if _instance is None:
        _instance = GeminiClient()
    return _instance
