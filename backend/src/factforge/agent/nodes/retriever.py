"""Node 2: Evidence retriever.

For each sub-claim, fetch web evidence via DuckDuckGo + optional page-text
enrichment. Outputs `sub_results` with `evidence` populated for each
sub-claim (scores remain empty until the NLI node runs).

Runs all sub-claim retrievals in parallel.
"""

from __future__ import annotations

import asyncio

import structlog

from factforge.agent.state import AgentState, SubClaimResult
from factforge.clients.search import retrieve_evidence

logger = structlog.get_logger(__name__)

# How many evidence snippets to retrieve per sub-claim. The downstream
# NLI cost scales linearly with this — 5 is a good balance for Gemini-Flash
# +DeBERTa.
EVIDENCE_K = 5


async def retriever_node(state: AgentState) -> dict:
    """Retrieve evidence for each sub-claim in parallel."""
    sub_claims = state.get("sub_claims") or []
    if not sub_claims:
        return {"sub_results": []}

    evidence_lists = await asyncio.gather(
        *(retrieve_evidence(sc, k=EVIDENCE_K, enrich=True) for sc in sub_claims),
        return_exceptions=False,
    )

    sub_results: list[SubClaimResult] = []
    for sc, evidence in zip(sub_claims, evidence_lists, strict=True):
        sub_results.append(
            SubClaimResult(
                sub_claim=sc,
                evidence=evidence,
                scores=[],
                verdict="",
                confidence=0.0,
                probs={},
                reason="",
            )
        )

    logger.info(
        "retriever_done",
        n_sub_claims=len(sub_claims),
        evidence_counts=[len(e) for e in evidence_lists],
    )
    return {"sub_results": sub_results}
