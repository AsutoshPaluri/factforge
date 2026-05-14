"""Node 5: Plain-English summary of the verdict.

Reads the synthesized verdict + per-sub-claim NLI evidence and produces a
short, factual explanation a user can actually understand. Cites specific
named sources (e.g. "MIT McGovern Institute", "Britannica") so the user
can trust + audit the verdict.

This is the bridge between the technical pipeline (probabilities, scores,
weighted aggregation) and a user-readable answer.
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel, Field

from factforge.agent.state import AgentState
from factforge.clients.gemini import get_gemini

logger = structlog.get_logger(__name__)

_MAX_EVIDENCE_PER_SUB = 3
_MAX_SNIPPET_CHARS = 350


class _SummaryOutput(BaseModel):
    """Structured output schema enforced via Gemini's response_schema."""

    summary: str = Field(
        description=(
            "2-3 paragraph plain-English explanation of the verdict. "
            "Cite specific named sources by name (e.g. 'NASA', 'Britannica', "
            "'MIT McGovern Institute'). Factual, neutral tone. No bullet "
            "points - write as flowing prose. Target 150-300 words total."
        ),
        min_length=80,
        max_length=2500,
    )


def _format_evidence_block(sub_results: list) -> str:
    """Compose a compact evidence digest for the LLM prompt."""
    lines: list[str] = []
    for sr in sub_results:
        verdict = sr.get("verdict", "")
        confidence = sr.get("confidence", 0.0)
        lines.append(
            f"\n## Sub-claim: \"{sr.get('sub_claim', '')}\""
        )
        lines.append(
            f"   Verdict: {verdict} ({confidence:.0%} confidence)"
        )

        evidence = sr.get("evidence", []) or []
        scores = sr.get("scores", []) or []
        for i, (ev, sc) in enumerate(
            zip(evidence[:_MAX_EVIDENCE_PER_SUB], scores[:_MAX_EVIDENCE_PER_SUB])
        ):
            snippet = ev.snippet[:_MAX_SNIPPET_CHARS]
            if len(ev.snippet) > _MAX_SNIPPET_CHARS:
                snippet = snippet.rstrip() + "..."
            lines.append(f"   Evidence #{i + 1}:")
            lines.append(f"   - Source: {ev.title}")
            lines.append(f"   - URL: {ev.url}")
            lines.append(
                f"   - NLI: entail={sc.entailment:.0%} / "
                f"neutral={sc.neutral:.0%} / contra={sc.contradiction:.0%}"
            )
            lines.append(f"   - Excerpt: {snippet}")
    return "\n".join(lines)


_PROMPT = """\
You are writing a brief, factual explanation of a fact-check verdict for the
user who submitted the claim. They want to understand WHY this verdict and
WHAT the evidence actually says.

Original claim: "{claim}"
Final verdict: {verdict} ({confidence:.0%} confidence)

Evidence retrieved and scored against the claim:
{evidence_block}

Write a 2-3 paragraph explanation:
- Paragraph 1: State the verdict and the most important reason.
- Paragraph 2: What the evidence actually says. Cite specific NAMED sources
  (e.g. "MIT McGovern Institute", "NASA", "Britannica", "the CDC"). Briefly
  quote or paraphrase the most informative excerpts.
- Paragraph 3 (optional, only if useful): Origin of the claim, why it
  persists, or context the reader should know.

Hard rules:
- Tone is factual and neutral. No editorializing, no moralizing.
- No bullet points; flow as prose.
- No "as an AI" or meta-commentary about the verdict process.
- Do NOT make up sources or citations - only use what's in the evidence above.
- Target length: 150-300 words.
"""


async def summarizer_node(state: AgentState) -> dict:
    """Generate a plain-English explanation of the verdict."""
    claim = state.get("claim") or ""
    verdict = state.get("final_verdict") or ""
    confidence = state.get("final_confidence") or 0.0
    sub_results = state.get("sub_results") or []

    if not sub_results or not verdict:
        return {"summary": ""}

    # Build context block
    evidence_block = _format_evidence_block(sub_results)

    # If there's no claim text (image-only input), use the first sub-claim
    # as the displayable claim
    display_claim = claim
    if not display_claim and sub_results:
        display_claim = sub_results[0].get("sub_claim", "")

    prompt = _PROMPT.format(
        claim=display_claim,
        verdict=verdict,
        confidence=confidence,
        evidence_block=evidence_block,
    )

    try:
        gemini = get_gemini()
        result = await gemini.generate_structured(prompt, _SummaryOutput)
        summary = result.summary.strip()
    except Exception as e:
        # Don't fail the whole pipeline if the summary call has trouble —
        # the verdict + evidence are still useful on their own.
        logger.warning("summarizer_failed", error=str(e))
        summary = ""

    logger.info("summarizer_done", chars=len(summary))
    return {"summary": summary}
