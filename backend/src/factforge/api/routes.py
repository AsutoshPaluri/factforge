"""Public HTTP routes for factforge.

Exposes the LangGraph agent as a single POST endpoint so the frontend
can submit claims (text + optional image) and receive verdicts.
"""

from __future__ import annotations

import base64
import binascii
import time

import structlog
from fastapi import APIRouter, HTTPException

from factforge.agent.graph import get_graph
from factforge.api.schemas import (
    ClaimRequest,
    ClaimResponse,
    EvidenceOut,
    SubClaimOut,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["claims"])


def _build_sub_claim_out(sr: dict) -> SubClaimOut:
    """Convert internal SubClaimResult dict into the API-facing schema."""
    evidence = sr.get("evidence", []) or []
    scores = sr.get("scores", []) or []

    # Top-3 evidence by entailment+contradiction (the "informativeness" signal).
    paired = list(zip(evidence, scores, strict=False))
    paired.sort(
        key=lambda t: (t[1].entailment + t[1].contradiction) if t[1] else 0.0,
        reverse=True,
    )
    top = paired[:3]

    return SubClaimOut(
        sub_claim=sr.get("sub_claim", ""),
        verdict=sr.get("verdict", ""),
        confidence=sr.get("confidence", 0.0),
        probs=sr.get("probs", {}),
        reason=sr.get("reason", ""),
        evidence_count=len(evidence),
        top_evidence=[
            EvidenceOut(
                title=ev.title,
                url=ev.url,
                snippet=ev.snippet,
                source_is_fulltext=ev.source_is_fulltext,
                entailment=sc.entailment,
                neutral=sc.neutral,
                contradiction=sc.contradiction,
            )
            for ev, sc in top
            if sc is not None
        ],
    )


@router.post("/claims", response_model=ClaimResponse)
async def submit_claim(payload: ClaimRequest) -> ClaimResponse:
    """Fact-check a claim end-to-end.

    Accepts text claim, image claim (via base64), or both. Pipeline:
    decompose (Gemini, vision-capable) → retrieve (DDG) → NLI (DeBERTa) →
    synthesize.
    """
    t0 = time.time()
    graph = get_graph()

    # Decode the optional image payload
    image_bytes: bytes | None = None
    if payload.image_b64:
        try:
            image_bytes = base64.b64decode(payload.image_b64, validate=True)
        except (ValueError, binascii.Error) as e:
            raise HTTPException(
                status_code=400,
                detail=f"invalid base64 image: {e}",
            ) from e

    agent_input = {
        "claim": payload.claim.strip(),
        "image_bytes": image_bytes,
        "image_mime": payload.image_mime,
    }

    try:
        final_state = await graph.ainvoke(agent_input)
    except Exception as e:
        logger.exception(
            "claim_pipeline_failed",
            claim=payload.claim[:80],
            had_image=image_bytes is not None,
        )
        raise HTTPException(
            status_code=500,
            detail=f"agent pipeline error: {type(e).__name__}",
        ) from e

    duration_ms = int((time.time() - t0) * 1000)

    sub_results_raw = final_state.get("sub_results") or []
    sub_results_out = [_build_sub_claim_out(sr) for sr in sub_results_raw]

    logger.info(
        "claim_completed",
        claim=payload.claim[:80],
        had_image=image_bytes is not None,
        verdict=final_state.get("final_verdict"),
        confidence=final_state.get("final_confidence"),
        duration_ms=duration_ms,
    )

    return ClaimResponse(
        claim=payload.claim,
        sub_claims=final_state.get("sub_claims", []),
        verdict=final_state["final_verdict"],
        confidence=final_state["final_confidence"],
        probs=final_state["final_probs"],
        reason=final_state.get("final_reason", ""),
        summary=final_state.get("summary", ""),
        sub_results=sub_results_out,
        duration_ms=duration_ms,
        was_multimodal=image_bytes is not None,
    )
