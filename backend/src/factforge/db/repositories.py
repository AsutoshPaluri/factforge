"""Database access layer for `claims` and `feedback` tables.

Single point of contact between the rest of the app and Supabase.
All CRUD goes through these functions — nothing else imports the
Supabase client directly. That way DB details (table names, column
shapes, error handling) stay encapsulated here.

Functions return Python-native types (dicts, str, etc.) so callers
don't need to know about supabase-py response objects.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import structlog

from factforge.db.supabase_client import get_supabase, is_configured

logger = structlog.get_logger(__name__)

CLAIMS_TABLE = "claims"
FEEDBACK_TABLE = "feedback"


# ---------------------------------------------------------------------------
# claims
# ---------------------------------------------------------------------------
async def save_claim(
    *,
    claim: str,
    was_multimodal: bool,
    sub_claims: list[str],
    verdict: str,
    confidence: float,
    probs: dict[str, float],
    reason: str,
    summary: str,
    sub_results: list[dict[str, Any]],
    duration_ms: int,
    parent_claim_id: str | None = None,
    feedback_used: str | None = None,
    claim_embedding: list[float] | None = None,
) -> str:
    """Persist a verdict and return its UUID as a string.

    `claim_embedding` should be a 384-dim L2-normalised vector (the
    output of clients/embeddings.py). Storing it lets future similar
    claims look up this row + its feedback via pgvector.

    Raises RuntimeError if Supabase is unreachable.
    """
    if not is_configured():
        raise RuntimeError("Supabase not configured")

    row: dict[str, Any] = {
        "claim": claim,
        "was_multimodal": was_multimodal,
        "sub_claims": sub_claims,
        "verdict": verdict,
        "confidence": confidence,
        "probs": probs,
        "reason": reason,
        "summary": summary,
        "sub_results": sub_results,
        "duration_ms": duration_ms,
        "parent_claim_id": parent_claim_id,
        "feedback_used": feedback_used,
    }
    if claim_embedding is not None:
        row["claim_embedding"] = claim_embedding

    sb = get_supabase()
    res = sb.table(CLAIMS_TABLE).insert(row).execute()

    if not res.data:
        raise RuntimeError(f"save_claim: empty insert response: {res}")

    claim_id = res.data[0]["id"]
    logger.info(
        "claim_saved",
        claim_id=claim_id,
        verdict=verdict,
        is_refinement=parent_claim_id is not None,
        has_embedding=claim_embedding is not None,
    )
    return claim_id


async def find_similar_past_feedback(
    embedding: list[float],
    k: int = 5,
    min_similarity: float = 0.75,
) -> list[dict[str, Any]]:
    """Return bad-feedback comments from past claims similar to this embedding.

    Uses pgvector cosine distance (`<=>` operator) via an RPC, since the
    supabase-py client doesn't natively expose vector operators. Falls
    back to in-Python similarity if the RPC isn't set up.

    Args:
        embedding: 384-dim vector for the new claim
        k: max number of past claims to consider
        min_similarity: cosine similarity threshold (0..1) — only return
            matches above this. Cosine distance = 1 - similarity, so we
            convert internally.

    Returns:
        List of dicts: [{claim, comment, similarity}, ...] ordered by
        descending similarity. Empty if no matches above threshold.
    """
    if not is_configured():
        return []

    sb = get_supabase()

    # Compose a SQL filter that pgvector understands. We use the RPC
    # interface so we can return both the claim text AND the joined
    # feedback comment in one round-trip.
    #
    # NOTE: this requires a SQL function `match_claim_feedback` in the DB.
    # If it doesn't exist yet, fall back to a (slower) two-query
    # in-Python join below.
    try:
        res = sb.rpc(
            "match_claim_feedback",
            {
                "query_embedding": embedding,
                "match_threshold": min_similarity,
                "match_count": k,
            },
        ).execute()
        rows = res.data or []
        return [
            {
                "claim": r["claim"],
                "comment": r["comment"],
                "similarity": r["similarity"],
            }
            for r in rows
        ]
    except Exception as e:
        # RPC not installed yet — fall back to in-Python similarity.
        logger.debug(
            "match_claim_feedback_rpc_unavailable_falling_back", error=str(e)
        )
        return await _python_similarity_fallback(embedding, k, min_similarity)


async def _python_similarity_fallback(
    embedding: list[float],
    k: int,
    min_similarity: float,
) -> list[dict[str, Any]]:
    """Pull recent claims + their bad feedback, compute cosine in Python.

    Scales poorly past ~10k claims but is correct and zero-setup. We can
    promote to the SQL RPC version later — see commented function below.
    """
    sb = get_supabase()
    # Pull most recent ~200 claims with embeddings. For portfolio traffic
    # that's plenty.
    rows_res = (
        sb.table(CLAIMS_TABLE)
        .select("id, claim, claim_embedding")
        .not_.is_("claim_embedding", "null")
        .order("created_at", desc=True)
        .limit(200)
        .execute()
    )
    candidates = rows_res.data or []
    if not candidates:
        return []

    # Compute cosine similarity (embeddings are L2-normalised, so dot product)
    import numpy as np

    q = np.asarray(embedding, dtype=np.float32)
    scored: list[tuple[float, dict]] = []
    for row in candidates:
        emb = row.get("claim_embedding")
        if not emb:
            continue
        # supabase-py returns pgvector as a string like "[0.1, 0.2, ...]"
        if isinstance(emb, str):
            try:
                import json as _json

                emb = _json.loads(emb)
            except Exception:
                continue
        v = np.asarray(emb, dtype=np.float32)
        if v.shape != q.shape:
            continue
        sim = float(np.dot(q, v))
        if sim >= min_similarity:
            scored.append((sim, row))

    if not scored:
        return []

    scored.sort(key=lambda t: -t[0])
    top_claim_ids = [row["id"] for _, row in scored[:k]]

    # Pull associated bad-feedback comments for those claims
    fb_res = (
        sb.table(FEEDBACK_TABLE)
        .select("claim_id, comment")
        .in_("claim_id", top_claim_ids)
        .eq("kind", "bad")
        .not_.is_("comment", "null")
        .execute()
    )
    feedback_by_claim: dict[str, list[str]] = {}
    for fb in fb_res.data or []:
        cid = fb["claim_id"]
        feedback_by_claim.setdefault(cid, []).append(fb["comment"])

    # Stitch
    result: list[dict[str, Any]] = []
    for sim, row in scored[:k]:
        comments = feedback_by_claim.get(row["id"], [])
        if not comments:
            continue
        for c in comments:
            result.append(
                {"claim": row["claim"], "comment": c, "similarity": sim}
            )

    return result


async def get_claim_by_id(claim_id: str) -> dict | None:
    """Fetch one claim row by UUID. None if not found."""
    if not is_configured():
        raise RuntimeError("Supabase not configured")

    # Validate UUID format defensively — Supabase will 400 otherwise.
    try:
        UUID(claim_id)
    except (ValueError, TypeError):
        return None

    sb = get_supabase()
    res = (
        sb.table(CLAIMS_TABLE)
        .select("*")
        .eq("id", claim_id)
        .limit(1)
        .execute()
    )

    if not res.data:
        return None
    return res.data[0]


# ---------------------------------------------------------------------------
# feedback
# ---------------------------------------------------------------------------
async def save_feedback(
    *,
    claim_id: str,
    kind: str,
    comment: str | None = None,
    user_ip_hash: str | None = None,
) -> str:
    """Persist one feedback row and return its UUID.

    Raises ValueError if `kind` isn't 'good'/'bad' (CHECK constraint
    would reject it server-side anyway, but failing fast is cheaper).
    """
    if kind not in ("good", "bad"):
        raise ValueError(f"feedback.kind must be 'good' or 'bad', got {kind!r}")

    if not is_configured():
        raise RuntimeError("Supabase not configured")

    row = {
        "claim_id": claim_id,
        "kind": kind,
        "comment": comment,
        "user_ip_hash": user_ip_hash,
    }

    sb = get_supabase()
    res = sb.table(FEEDBACK_TABLE).insert(row).execute()

    if not res.data:
        raise RuntimeError(f"save_feedback: empty insert response: {res}")

    fb_id = res.data[0]["id"]
    logger.info("feedback_saved", feedback_id=fb_id, claim_id=claim_id, kind=kind)
    return fb_id
