"""Pydantic request / response schemas for the public API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["Real", "Misinformation", "Disinformation"]


class ClaimRequest(BaseModel):
    """POST /api/v1/claims body."""

    claim: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The claim to fact-check. Plain English, one or more sentences.",
    )
    # Image input is supported by the agent but not yet exposed in the API.
    # Add `image_b64: str | None = None` here when we wire up multimodal UI.


class EvidenceOut(BaseModel):
    """One retrieved evidence snippet with its NLI scores."""

    title: str
    url: str
    snippet: str
    source_is_fulltext: bool
    entailment: float
    neutral: float
    contradiction: float


class SubClaimOut(BaseModel):
    """Per-sub-claim result block."""

    sub_claim: str
    verdict: Verdict | Literal[""]
    confidence: float
    probs: dict[str, float]
    reason: str
    evidence_count: int
    top_evidence: list[EvidenceOut] = Field(
        default_factory=list,
        description="Top-3 most informative snippets (sorted by info_score).",
    )


class ClaimResponse(BaseModel):
    """POST /api/v1/claims response body."""

    claim: str
    sub_claims: list[str]
    verdict: Verdict
    confidence: float
    probs: dict[str, float]
    reason: str
    sub_results: list[SubClaimOut]
    duration_ms: int
