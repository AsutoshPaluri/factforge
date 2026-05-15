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

from factforge.agent.state import AgentState
from factforge.clients.groq import get_groq
from factforge.config import settings

logger = structlog.get_logger(__name__)

_MAX_EVIDENCE_PER_SUB = 3
_MAX_SNIPPET_CHARS = 350


# Plain-text summary — markdown rendered client-side. Avoiding JSON-mode
# because multi-line markdown with asterisks, quotes, and bullets is brittle
# inside escaped JSON strings; some completions return invalid JSON and
# blow up Pydantic parsing.


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
You are writing a fact-check explanation in the style of Google's AI Overview
— scannable, with a clear lead, bulleted key reasons, and named sources.

Claim: "{claim}"
Verdict: {verdict} ({confidence:.0%} confidence)

Verdict vocabulary (use these terms in your writing):
  - "Credible"       — the claim is supported by the evidence
  - "Not Credible"   — the claim is refuted by the evidence
  - "Uncertain"      — evidence is mixed, sparse, or genuinely ambiguous;
                       agent abstains from a binary call

Evidence retrieved and scored against the claim:
{evidence_block}

OUTPUT FORMAT (markdown — render exactly this structure):

1. LEAD PARAGRAPH (1-2 sentences):
   - State the verdict directly using the vocabulary above. Reference 1-2
     named sources from the evidence (e.g. "According to *NASA*..." or
     "*Britannica* documents that...").
   - A reader should know the answer from this paragraph alone.

2. EMPTY LINE, then "Key reasons:" on its own line, then EMPTY LINE.

3. 3-5 BULLETS. Each bullet:
   - starts with "- "
   - opens with a **bold lead-in** of 1-4 words followed by ":"
   - then 1-2 sentence explanation
   - cite a specific named source where relevant (e.g. "per *NASA*",
     "*MIT McGovern Institute* notes...")

4. OPTIONAL CLOSING PARAGRAPH (1-2 sentences). Only if it adds real
   value — origin of the misconception, why it persists, when first
   disproven, OR (if Uncertain) what evidence would clarify the verdict.
   Skip if not relevant.

EXAMPLE — for a "Not Credible" verdict on "Earth is flat":

The claim that Earth is flat is not credible. *NASA* and *Britannica*
both document that Earth is an oblate spheroid, slightly bulged at the
equator, as confirmed by direct observation from space.

Key reasons:

- **Ships over the horizon:** As ships sail away they disappear from the bottom up — a visible sign of Earth's curvature noted since antiquity.
- **Lunar eclipses:** Earth casts a circular shadow on the Moon during every lunar eclipse, geometrically only possible for a sphere.
- **Satellite imagery:** Tens of thousands of satellites and direct photographs from space confirm Earth's spherical shape, per *NASA*.

The flat-Earth idea persists today primarily as conspiracy content
despite being scientifically disproven for over two millennia.

EXAMPLE — for an "Uncertain" verdict:

The claim "[X]" is **uncertain** based on the retrieved evidence — sources
either don't directly address the assertion or split between supporting
and refuting it.

Key reasons:

- **Mixed signals:** *Source A* supports the claim while *Source B* explicitly contradicts it.
- **Sparse coverage:** Only 2 of 8 retrieved sources directly addressed the assertion.
- **Definitional ambiguity:** The claim depends on how "[term]" is defined, which varies across sources.

A clearer verdict would require sources that directly address the
specific assertion rather than adjacent topics.

RULES:
- Cite REAL named sources from the evidence above. Never invent.
- Factual, neutral tone. No moralizing, no exclamations.
- No "as an AI" or meta-commentary.
- Use only "Credible / Not Credible / Uncertain" — NOT "Real",
  "Disinformation", "Misinformation", "fake news", etc.
- Plain markdown only. No code fences, no headings (no ###).
- 150-300 words total.
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
        groq = get_groq()
        # Use the lighter 8b model for the summary — 5x more daily-token
        # headroom on Groq free tier (500k TPD vs 100k for 70b), and it
        # handles 2-3 paragraph markdown narrative just fine.
        # temperature=0.1 keeps wording stable across reruns (consistency
        # for the same claim) without going fully deterministic (which can
        # produce stilted phrasing).
        summary = (
            await groq.generate_text(
                prompt,
                model=settings.groq_summary_model,
                temperature=0.1,
            )
        ).strip()
        # Strip any stray markdown code-fence the model might add
        if summary.startswith("```"):
            summary = summary.split("```")[1]
            if summary.startswith("markdown"):
                summary = summary[len("markdown") :]
            summary = summary.strip()
    except Exception as e:
        # Don't fail the whole pipeline if the summary call has trouble —
        # the verdict + evidence are still useful on their own.
        logger.warning("summarizer_failed", error=str(e))
        summary = ""

    logger.info(
        "summarizer_done", chars=len(summary), model=settings.groq_summary_model
    )
    return {"summary": summary}
