"""Node 1: Claim decomposer.

Breaks a compound claim into atomic, individually-verifiable sub-claims.

Routing:
  - Text-only claim -> Groq (Llama 3.3 70B, fast + generous free tier)
  - Image present  -> Gemini (vision-capable; Groq doesn't support images yet)

Examples:
    "Vaccines cause autism and the earth is flat"
        -> ["Vaccines cause autism", "The earth is flat"]
    "Drinking water lowers your IQ"
        -> ["Drinking water lowers your IQ"]   (already atomic)
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel, Field

from factforge.agent.state import AgentState
from factforge.clients.gemini import get_gemini
from factforge.clients.groq import get_groq

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

_PROMPT_WITH_FEEDBACK_TEMPLATE = """\
You are RE-decomposing a claim into atomic, individually-verifiable sub-claims,
based on user feedback about a previous attempt.

Same rules as before:
1. Each sub-claim must be a SINGLE factual assertion.
2. Each sub-claim must be SELF-CONTAINED.
3. Strip rhetorical framing.
4. If the input is atomic, return one element.
5. NEVER return an empty list.

CRITICAL RULE — DO NOT VIOLATE:
Every sub-claim must be a POSITIVE STATEMENT of the original claim's assertion.
You are decomposing what the claim CLAIMS to be true. You are NOT writing
refutations, negations, or counter-claims. The downstream verifier handles
deciding if each sub-claim is true or false — your job is only to break down
what's being asserted.

Examples for a claim "Earth is flat":
  GOOD sub-claim: "Earth has a flat shape"
  GOOD sub-claim: "Earth is not a sphere"  (still a statement OF the claim)
  GOOD sub-claim: "Photos from space showing a flat Earth exist"
  BAD  sub-claim: "Earth is an oblate spheroid"  (this is a REFUTATION,
                                                  not part of the original claim)
  BAD  sub-claim: "Scientific consensus says Earth is round"  (refutation)
  BAD  sub-claim: "There is no evidence Earth is flat"  (negation of the claim)

USER FEEDBACK:
The user gave feedback on a previous attempt:
"{feedback}"

Use the feedback to choose BETTER FRAMINGS of the original claim — different
angles, dimensions, or testable aspects — but every sub-claim must STILL
assert what the original claim is asserting. If feedback says "consider
peer-reviewed sources", you might phrase a sub-claim as "Peer-reviewed
journals have published evidence supporting [the claim]" (testable; downstream
NLI decides if true). You do NOT write "peer-reviewed journals refute X".

Don't echo the feedback verbatim. Just produce better-targeted sub-claims
that still assert the claim.

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


def _build_text_prompt(
    claim: str, feedback: str | None, learned_feedback: str | None
) -> str:
    """Build the decomposer prompt with optional feedback + learned context."""
    if feedback or learned_feedback:
        # When ANY contextual feedback is present, use the constrained
        # prompt that prevents the LLM from reframing the claim into its
        # negation.
        feedback_section = (
            f'User feedback on a previous attempt for this claim:\n"{feedback}"\n'
            if feedback
            else ""
        )
        learned_section = (
            f"Common pitfalls from past users on similar claims (use these as guidance):\n"
            f"{learned_feedback}\n"
            if learned_feedback
            else ""
        )
        return _PROMPT_WITH_FEEDBACK_TEMPLATE.format(
            claim=claim,
            feedback=(
                (feedback_section + learned_section).strip()
                or "(no specific feedback)"
            ),
        )
    return _PROMPT_TEMPLATE.format(claim=claim)


async def decomposer_node(state: AgentState) -> dict:
    """Decompose `state['claim']` (and optional image) into sub-claims.

    Context-aware:
      - `state['feedback']`        — current user's feedback on a previous
                                     verdict (from /refine).
      - `state['learned_feedback']`— aggregated feedback from past users
                                     on similar claims (from pgvector
                                     lookup in routes.py).
    Either / both / neither can be present.
    """
    claim = state.get("claim", "").strip()
    image_bytes = state.get("image_bytes")
    image_mime = state.get("image_mime") or "image/jpeg"
    feedback = (state.get("feedback") or "").strip() or None
    learned = (state.get("learned_feedback") or "").strip() or None

    if not claim and not image_bytes:
        raise ValueError("decomposer: at least one of claim or image_bytes required")

    if image_bytes:
        # Vision: route to Gemini (Groq doesn't support image input yet)
        text_part = (
            f"Accompanying text:\n{claim}" if claim else "No accompanying text."
        )
        if feedback:
            text_part += (
                f"\n\nUser gave this feedback on a previous attempt; "
                f"factor it in: {feedback}"
            )
        if learned:
            text_part += (
                f"\n\nCommon pitfalls from past users on similar claims:\n{learned}"
            )
        prompt = _MULTIMODAL_PROMPT_TEMPLATE.format(text_part=text_part)
        gemini = get_gemini()
        result = await gemini.generate_multimodal(
            prompt=prompt,
            image_bytes=image_bytes,
            image_mime=image_mime,
            schema=_DecomposerOutput,
        )
        llm_used = "gemini"
    else:
        prompt = _build_text_prompt(claim, feedback, learned)
        groq = get_groq()
        result = await groq.generate_structured(prompt, _DecomposerOutput)
        llm_used = "groq"

    sub_claims = [c.strip() for c in result.sub_claims if c and c.strip()]
    if not sub_claims:
        sub_claims = [claim] if claim else ["[image-only claim]"]

    logger.info(
        "decomposer_done",
        original_claim=claim[:80],
        n_sub_claims=len(sub_claims),
        llm=llm_used,
        had_feedback=feedback is not None,
        had_learned_feedback=learned is not None,
    )
    return {"sub_claims": sub_claims}
