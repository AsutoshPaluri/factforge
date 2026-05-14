"""Node 3: NLI verifier.

For each sub-claim's evidence, score (snippet -> sub-claim) with the
embedded DeBERTa-MNLI-FEVER model. Populates `scores` on each
SubClaimResult.

Uses the perception-verb-normalised form of the sub-claim as the NLI
hypothesis so "Earth looks flat" and "Earth is flat" get equivalent scores.
"""

from __future__ import annotations

import structlog

from factforge.agent.state import AgentState, SubClaimResult
from factforge.clients.nli import get_verifier
from factforge.utils.text import normalise_perception_verbs

logger = structlog.get_logger(__name__)


async def nli_node(state: AgentState) -> dict:
    """Score each (evidence, sub_claim) pair with the NLI model."""
    sub_results = state.get("sub_results") or []
    if not sub_results:
        return {"sub_results": []}

    verifier = await get_verifier()

    updated: list[SubClaimResult] = []
    for sr in sub_results:
        evidence = sr["evidence"]
        if not evidence:
            updated.append({**sr, "scores": []})
            continue

        hypothesis = normalise_perception_verbs(sr["sub_claim"])
        premises = [e.snippet for e in evidence]
        scores = await verifier.score(premises, hypothesis)

        updated.append({**sr, "scores": scores})

    logger.info(
        "nli_done",
        n_sub_claims=len(updated),
        score_counts=[len(sr["scores"]) for sr in updated],
    )
    return {"sub_results": updated}
