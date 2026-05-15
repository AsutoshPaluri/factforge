"""Public HTTP routes for factforge.

Endpoints:
  POST /api/v1/claims                       — submit claim, get verdict
  POST /api/v1/claims/{id}/feedback         — thumbs/comment on a verdict
  POST /api/v1/claims/{id}/refine           — re-run agent with feedback
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import time

import structlog
from fastapi import APIRouter, HTTPException, Request

from factforge.agent.graph import get_graph
from factforge.api.schemas import (
    ClaimRequest,
    ClaimResponse,
    EvidenceOut,
    FeedbackRequest,
    FeedbackResponse,
    RefineRequest,
    SubClaimOut,
)
from factforge.clients.embeddings import get_embedder
from factforge.clients.groq import get_groq
from factforge.config import settings
from factforge.db import repositories
from factforge.db.supabase_client import is_configured as db_configured

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["claims"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _client_ip(request: Request) -> str:
    """Best-effort IP. X-Forwarded-For first (for proxies), then peer."""
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    real = request.headers.get("x-real-ip", "")
    if real:
        return real.strip()
    return request.client.host if request.client else ""


def _hash_ip(ip: str | None) -> str | None:
    """SHA-256(ip)[:16] — opaque token for soft per-IP grouping, not ID."""
    if not ip:
        return None
    return hashlib.sha256(ip.encode("utf-8")).hexdigest()[:16]


def _build_sub_claim_out(sr: dict) -> SubClaimOut:
    """Convert internal SubClaimResult dict into the API-facing schema."""
    evidence = sr.get("evidence", []) or []
    scores = sr.get("scores", []) or []

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


async def _persist_claim(
    *,
    payload_claim: str,
    final_state: dict,
    duration_ms: int,
    was_multimodal: bool,
    sub_results_serialised: list[dict],
    parent_claim_id: str | None = None,
    feedback_used: str | None = None,
    claim_embedding: list[float] | None = None,
) -> str | None:
    """Save the verdict to Supabase. Returns claim_id or None on failure."""
    if not db_configured():
        logger.warning("supabase_not_configured_skipping_persist")
        return None
    try:
        return await repositories.save_claim(
            claim=payload_claim,
            was_multimodal=was_multimodal,
            sub_claims=final_state.get("sub_claims", []),
            verdict=final_state["final_verdict"],
            confidence=final_state["final_confidence"],
            probs=final_state["final_probs"],
            reason=final_state.get("final_reason", ""),
            summary=final_state.get("summary", ""),
            sub_results=sub_results_serialised,
            duration_ms=duration_ms,
            parent_claim_id=parent_claim_id,
            feedback_used=feedback_used,
            claim_embedding=claim_embedding,
        )
    except Exception as e:
        logger.exception("save_claim_failed", error=str(e))
        return None


async def _gather_learned_feedback(claim_text: str) -> tuple[list[float] | None, str | None]:
    """Embed the claim + retrieve aggregated past feedback on similar claims.

    Returns (embedding, learned_feedback_block). Either can be None on failure
    — agent degrades gracefully (runs without learned context).
    """
    if not db_configured() or not claim_text.strip():
        return None, None

    try:
        embedder = get_embedder()
        embedding = await embedder.embed_one(claim_text)
    except Exception as e:
        logger.warning("embedding_failed", error=str(e))
        return None, None

    try:
        matches = await repositories.find_similar_past_feedback(
            embedding=embedding, k=5, min_similarity=0.75
        )
    except Exception as e:
        logger.warning("find_similar_failed", error=str(e))
        return embedding, None

    if not matches:
        return embedding, None

    # Format the learned-feedback block for the decomposer prompt.
    lines = []
    for m in matches:
        comment = (m.get("comment") or "").strip()
        if not comment:
            continue
        prev_claim = (m.get("claim") or "").strip()[:60]
        sim_pct = int((m.get("similarity") or 0.0) * 100)
        lines.append(
            f'- "{comment}" (from similar claim "{prev_claim}", {sim_pct}% similar)'
        )
    learned = "\n".join(lines) if lines else None

    logger.info(
        "learned_feedback_gathered",
        n_matches=len(matches),
        n_used=len(lines) if learned else 0,
        claim=claim_text[:60],
    )
    return embedding, learned


def _sub_results_to_dicts(sub_results_out: list[SubClaimOut]) -> list[dict]:
    """For DB persistence — Pydantic models to plain dicts JSONB can store."""
    return [sr.model_dump() for sr in sub_results_out]


async def _validate_feedback_relevance(claim_text: str, feedback: str) -> bool:
    """Groq-judged relevance check. True if feedback is on-topic for the claim."""
    prompt = f"""You are checking if user feedback is relevant to a fact-check claim.

CLAIM: "{claim_text}"

USER FEEDBACK: "{feedback}"

Is the feedback ACTUALLY ABOUT this claim's fact-check verdict? Reply with a single word:

YES — if the feedback is about: accuracy of the verdict, quality of sources, \
missing context, interpretation of evidence, suggested sources to consider, \
or a substantive disagreement.

NO — if the feedback is: spam, random text, unrelated to the claim, a general \
opinion not actionable, or appears to be a different question entirely.

Reply with exactly YES or NO, nothing else."""
    try:
        groq = get_groq()
        reply = await groq.generate_text(prompt, temperature=0.0)
        return reply.strip().upper().startswith("YES")
    except Exception as e:
        logger.warning("feedback_relevance_check_failed", error=str(e))
        # Fail open — let the refinement proceed if the judge errors,
        # rather than block legitimate feedback.
        return True


# ---------------------------------------------------------------------------
# POST /claims — submit a new claim
# ---------------------------------------------------------------------------
@router.post("/claims", response_model=ClaimResponse)
async def submit_claim(payload: ClaimRequest) -> ClaimResponse:
    """Fact-check a claim end-to-end. Returns the verdict + a claim_id that
    can be used to submit feedback or trigger a refinement."""
    t0 = time.time()
    graph = get_graph()

    image_bytes: bytes | None = None
    if payload.image_b64:
        try:
            image_bytes = base64.b64decode(payload.image_b64, validate=True)
        except (ValueError, binascii.Error) as e:
            raise HTTPException(
                status_code=400, detail=f"invalid base64 image: {e}"
            ) from e

    # Memory-augmented context: look up bad feedback from past similar claims
    # and inject as `learned_feedback` into the agent state.
    embedding, learned_feedback = await _gather_learned_feedback(payload.claim)

    agent_input = {
        "claim": payload.claim.strip(),
        "image_bytes": image_bytes,
        "image_mime": payload.image_mime,
        "learned_feedback": learned_feedback,
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
            status_code=500, detail=f"agent pipeline error: {type(e).__name__}"
        ) from e

    duration_ms = int((time.time() - t0) * 1000)
    sub_results_raw = final_state.get("sub_results") or []
    sub_results_out = [_build_sub_claim_out(sr) for sr in sub_results_raw]

    claim_id = await _persist_claim(
        payload_claim=payload.claim,
        final_state=final_state,
        duration_ms=duration_ms,
        was_multimodal=image_bytes is not None,
        sub_results_serialised=_sub_results_to_dicts(sub_results_out),
        claim_embedding=embedding,
    )

    logger.info(
        "claim_completed",
        claim=payload.claim[:80],
        had_image=image_bytes is not None,
        verdict=final_state.get("final_verdict"),
        confidence=final_state.get("final_confidence"),
        duration_ms=duration_ms,
        claim_id=claim_id,
    )

    return ClaimResponse(
        claim_id=claim_id,
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


# ---------------------------------------------------------------------------
# POST /claims/{id}/feedback — thumbs + optional comment
# ---------------------------------------------------------------------------
@router.post("/claims/{claim_id}/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    claim_id: str, payload: FeedbackRequest, request: Request
) -> FeedbackResponse:
    """Record user thumbs-up/down + optional comment on a verdict.

    Validates the claim_id exists before accepting feedback (so random
    UUIDs can't flood the table)."""
    if not db_configured():
        raise HTTPException(
            status_code=503, detail="Feedback requires Supabase to be configured."
        )

    claim_row = await repositories.get_claim_by_id(claim_id)
    if claim_row is None:
        raise HTTPException(status_code=404, detail="claim not found")

    try:
        fb_id = await repositories.save_feedback(
            claim_id=claim_id,
            kind=payload.kind,
            comment=(payload.comment or "").strip() or None,
            user_ip_hash=_hash_ip(_client_ip(request)),
        )
    except Exception as e:
        logger.exception("save_feedback_failed", error=str(e))
        raise HTTPException(
            status_code=500, detail="failed to save feedback"
        ) from e

    return FeedbackResponse(ok=True, feedback_id=fb_id)


# ---------------------------------------------------------------------------
# POST /claims/{id}/refine — re-run agent with user feedback
# ---------------------------------------------------------------------------
@router.post("/claims/{claim_id}/refine", response_model=ClaimResponse)
async def refine_claim(
    claim_id: str, payload: RefineRequest
) -> ClaimResponse:
    """Re-run the agent on the original claim with user feedback as context.

    Flow:
      1. Load original claim from DB.
      2. Groq-judge: is the feedback relevant?
      3. If yes, run the agent with `feedback` field in state.
         The decomposer factors the feedback into its sub-claim split;
         downstream nodes execute normally on the new sub-claims.
      4. Save the new verdict with parent_claim_id pointing back to original.
    """
    if not db_configured():
        raise HTTPException(
            status_code=503, detail="Refinement requires Supabase to be configured."
        )

    parent = await repositories.get_claim_by_id(claim_id)
    if parent is None:
        raise HTTPException(status_code=404, detail="original claim not found")

    parent_claim_text: str = parent["claim"]
    feedback = payload.feedback.strip()

    # Step 1: relevance check
    is_relevant = await _validate_feedback_relevance(parent_claim_text, feedback)
    if not is_relevant:
        raise HTTPException(
            status_code=400,
            detail=(
                "Your feedback doesn't appear to be about this claim. Please "
                "be specific — e.g. 'you missed peer-reviewed studies', "
                "'this verdict ignored X', or 'the source on point 2 is "
                "biased'."
            ),
        )

    # Step 2: re-run agent (with both current + learned feedback)
    embedding, learned_feedback = await _gather_learned_feedback(parent_claim_text)
    t0 = time.time()
    graph = get_graph()
    try:
        final_state = await graph.ainvoke(
            {
                "claim": parent_claim_text,
                "feedback": feedback,
                "learned_feedback": learned_feedback,
            }
        )
    except Exception as e:
        logger.exception("refine_pipeline_failed", claim_id=claim_id, error=str(e))
        raise HTTPException(
            status_code=500, detail=f"agent pipeline error: {type(e).__name__}"
        ) from e

    duration_ms = int((time.time() - t0) * 1000)
    sub_results_raw = final_state.get("sub_results") or []
    sub_results_out = [_build_sub_claim_out(sr) for sr in sub_results_raw]

    # Step 3: persist the refinement (with embedding for future similarity lookups)
    new_claim_id = await _persist_claim(
        payload_claim=parent_claim_text,
        final_state=final_state,
        duration_ms=duration_ms,
        was_multimodal=parent.get("was_multimodal", False),
        sub_results_serialised=_sub_results_to_dicts(sub_results_out),
        parent_claim_id=claim_id,
        feedback_used=feedback,
        claim_embedding=embedding,
    )

    logger.info(
        "claim_refined",
        original_claim_id=claim_id,
        new_claim_id=new_claim_id,
        original_verdict=parent.get("verdict"),
        new_verdict=final_state.get("final_verdict"),
        feedback=feedback[:120],
        duration_ms=duration_ms,
    )

    return ClaimResponse(
        claim_id=new_claim_id,
        parent_claim_id=claim_id,
        claim=parent_claim_text,
        sub_claims=final_state.get("sub_claims", []),
        verdict=final_state["final_verdict"],
        confidence=final_state["final_confidence"],
        probs=final_state["final_probs"],
        reason=final_state.get("final_reason", ""),
        summary=final_state.get("summary", ""),
        sub_results=sub_results_out,
        duration_ms=duration_ms,
        was_multimodal=parent.get("was_multimodal", False),
    )
