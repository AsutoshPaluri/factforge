"""Node 1: Claim decomposer.

Breaks a compound claim into atomic, individually-verifiable sub-claims.

Examples:
    "Vaccines cause autism and the earth is flat"
        -> ["Vaccines cause autism", "The earth is flat"]
    "Drinking water lowers your IQ"
        -> ["Drinking water lowers your IQ"]   (already atomic)

If `image_bytes` is present, the image is shown to Gemini alongside any
text claim so the model can extract claims depicted in the image
(screenshot of a tweet, infographic, etc.).
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel, Field

from factforge.agent.state import AgentState
from factforge.clients.gemini import get_gemini

logger = structlog.get_logger(__name__)


class _DecomposerOutput(BaseModel):
    """Structured output schema enforced via Gemini's response_schema."""

    sub_claims: list[str] = Field(
        description=(
            "Atomic, independently-verifiable factual assertions. "
            "If the input is already atomic, return a single-element list "
            "containing the original claim. Strip rhetorical framing "
            "('I heard that...', 'apparently...') but keep the factual "
            "substance intact. Never return an empty list."
        ),
        min_length=1,
        max_length=8,
    )


_PROMPT_TEMPLATE = """\
You are decomposing a claim into atomic, individually-verifiable sub-claims.

Rules:
1. Each sub-claim must be a SINGLE factual assertion (no "and"/"but"/"or" joining).
2. Each sub-claim must be SELF-CONTAINED (understandable without the others).
3. Strip rhetorical framing ("I heard that", "apparently", "they say").
4. If the input is already atomic, return it as a single-element list.
5. NEVER return an empty list. If the input is unverifiable opinion ("X is the best"),
   still wrap it as a single sub-claim and let the downstream verifier handle it.

Input claim:
{claim}
"""

_MULTIMODAL_PROMPT_TEMPLATE = """\
You are decomposing a claim into atomic, individually-verifiable sub-claims.

You are given an image (which may be a screenshot, infographic, chart, or
photo) and optionally some accompanying text. Extract the factual claim(s)
being made and decompose them.

Same rules as text mode:
1. Each sub-claim must be a SINGLE factual assertion.
2. Each sub-claim must be SELF-CONTAINED.
3. Strip rhetorical framing.
4. NEVER return an empty list.

{text_part}
"""


async def decomposer_node(state: AgentState) -> dict:
    """Decompose `state['claim']` (and optional image) into sub-claims."""
    claim = state.get("claim", "").strip()
    image_bytes = state.get("image_bytes")
    image_mime = state.get("image_mime") or "image/jpeg"

    if not claim and not image_bytes:
        raise ValueError("decomposer: at least one of claim or image_bytes required")

    gemini = get_gemini()

    if image_bytes:
        text_part = (
            f"Accompanying text:\n{claim}" if claim else "No accompanying text."
        )
        prompt = _MULTIMODAL_PROMPT_TEMPLATE.format(text_part=text_part)
        result = await gemini.generate_multimodal(
            prompt=prompt,
            image_bytes=image_bytes,
            image_mime=image_mime,
            schema=_DecomposerOutput,
        )
    else:
        prompt = _PROMPT_TEMPLATE.format(claim=claim)
        result = await gemini.generate_structured(prompt, _DecomposerOutput)

    # `result` is _DecomposerOutput here
    sub_claims = [c.strip() for c in result.sub_claims if c and c.strip()]
    if not sub_claims:
        # Fallback: use the original claim as the single sub-claim
        sub_claims = [claim] if claim else ["[image-only claim]"]

    logger.info(
        "decomposer_done",
        original_claim=claim[:80],
        n_sub_claims=len(sub_claims),
    )
    return {"sub_claims": sub_claims}
