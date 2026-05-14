"""Gemini client smoke tests — actually hit the live Gemini API.

Marked @pytest.mark.slow because they consume free-tier tokens.
Each test uses ~50-100 tokens, well under the daily 1M cap.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from factforge.clients.gemini import GeminiClient
from factforge.config import settings


def _has_gemini_key() -> bool:
    return bool(settings.gemini_api_key)


@pytest.mark.slow
@pytest.mark.skipif(not _has_gemini_key(), reason="GEMINI_API_KEY not set")
@pytest.mark.asyncio
async def test_gemini_text_roundtrip() -> None:
    client = GeminiClient()
    text = await client.generate_text("Reply with exactly one word: pong")
    assert "pong" in text.lower()

    # Token usage should be populated after a successful call
    assert client.last_usage is not None
    assert client.last_usage.prompt_tokens > 0
    assert client.last_usage.total_tokens > 0


@pytest.mark.slow
@pytest.mark.skipif(not _has_gemini_key(), reason="GEMINI_API_KEY not set")
@pytest.mark.asyncio
async def test_gemini_structured_output() -> None:
    """Verify structured-output mode returns a valid Pydantic model."""

    class SubClaims(BaseModel):
        sub_claims: list[str] = Field(
            description="Atomic, individually-verifiable claims"
        )

    client = GeminiClient()
    result = await client.generate_structured(
        prompt=(
            "Decompose this compound claim into atomic sub-claims, one per "
            "verifiable assertion:\n\n"
            "'Vaccines cause autism and the earth is flat.'"
        ),
        schema=SubClaims,
    )

    assert isinstance(result, SubClaims)
    assert len(result.sub_claims) >= 2
    joined = " ".join(result.sub_claims).lower()
    assert "vaccin" in joined or "autism" in joined
    assert "earth" in joined or "flat" in joined
