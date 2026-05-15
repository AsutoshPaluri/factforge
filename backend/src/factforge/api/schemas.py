"""Pydantic request / response schemas for the public API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Verdict = Literal["Credible", "Uncertain", "Not Credible"]

# Max base64 payload size for an image — ~4.4 MB raw = ~6 MB encoded.
# Bigger than this likely means a HEIC / RAW that won't be useful anyway.
_MAX_IMAGE_B64_CHARS = 6_000_000

_ALLOWED_MIME = frozenset(
    {"image/jpeg", "image/png", "image/webp", "image/gif"}
)


class ClaimRequest(BaseModel):
    """POST /api/v1/claims body.

    At least one of `claim` or `image_b64` must be provided. If both, the
    agent uses the image to extract the claim and the text as additional
    context (e.g. "the screenshot below + this person says X").
    """

    claim: str = Field(
        default="",
        max_length=1000,
        description=(
            "The claim to fact-check. Optional if `image_b64` is provided "
            "(the agent will extract the claim from the image)."
        ),
    )
    image_b64: str | None = Field(
        default=None,
        description=(
            "Optional base64-encoded image (JPEG/PNG/WebP/GIF). The agent "
            "will run Gemini Vision on it to extract and decompose the claim."
        ),
    )
    image_mime: str = Field(
        default="image/jpeg",
        description="MIME type of the image. Must be image/jpeg|png|webp|gif.",
    )

    @model_validator(mode="after")
    def _validate(self) -> "ClaimRequest":
        if not self.claim.strip() and not self.image_b64:
            raise ValueError(
                "at least one of `claim` (>=3 chars) or `image_b64` is required"
            )
        if self.claim.strip() and len(self.claim.strip()) < 3:
            raise ValueError("`claim` must be at least 3 characters if provided")
        if self.image_b64:
            if self.image_mime not in _ALLOWED_MIME:
                raise ValueError(
                    f"image_mime must be one of {sorted(_ALLOWED_MIME)}"
                )
            if len(self.image_b64) > _MAX_IMAGE_B64_CHARS:
                raise ValueError(
                    f"image too large (base64 payload > {_MAX_IMAGE_B64_CHARS} chars)"
                )
        return self


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

    claim_id: str | None = Field(
        default=None,
        description=(
            "UUID of this verdict in the claims table. Use this to POST "
            "feedback or refine. None if DB persistence failed (degrades "
            "gracefully — verdict still returned)."
        ),
    )
    parent_claim_id: str | None = Field(
        default=None,
        description=(
            "If this verdict is a refinement of a previous one, this is the "
            "original's claim_id. None for first-pass verdicts."
        ),
    )
    claim: str
    sub_claims: list[str]
    verdict: Verdict
    confidence: float
    probs: dict[str, float]
    reason: str
    summary: str = Field(
        default="",
        description=(
            "Plain-English 2-3 paragraph explanation of the verdict, "
            "citing named sources from the retrieved evidence. Empty "
            "string if the summarizer step failed."
        ),
    )
    sub_results: list[SubClaimOut]
    duration_ms: int
    was_multimodal: bool = Field(
        default=False,
        description="True if an image was provided as input.",
    )


# ---------------------------------------------------------------------------
# Feedback / refinement
# ---------------------------------------------------------------------------
class FeedbackRequest(BaseModel):
    """POST /api/v1/claims/{id}/feedback body."""

    kind: Literal["good", "bad"]
    comment: str | None = Field(default=None, max_length=500)


class FeedbackResponse(BaseModel):
    """POST /api/v1/claims/{id}/feedback response."""

    ok: bool
    feedback_id: str


class RefineRequest(BaseModel):
    """POST /api/v1/claims/{id}/refine body.

    `feedback` is the natural-language note from the user explaining what
    they think went wrong. The agent uses this to re-decompose + re-retrieve,
    producing a new verdict whose parent_claim_id points back to the
    original.
    """

    feedback: str = Field(
        min_length=10,
        max_length=500,
        description=(
            "What the user thinks the previous verdict got wrong, or what "
            "they want the agent to consider this time. Free-form, 10-500 "
            "chars."
        ),
    )
