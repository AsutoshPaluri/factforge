"""LangGraph state schema for the fact-checking agent.

State flows through the graph as a TypedDict. Each node returns a partial
dict that LangGraph merges into the running state. We keep the schema flat
and serialisation-friendly (no nested ORM objects); the database layer
turns this into rows later.
"""

from __future__ import annotations

from typing import TypedDict

from factforge.clients.nli import NLIScore
from factforge.clients.search import EvidenceSnippet


class SubClaimResult(TypedDict):
    """Per-sub-claim verification result.

    Each sub-claim is verified independently against its own retrieved
    evidence; the synthesizer combines these into the final verdict.
    """

    sub_claim: str
    evidence: list[EvidenceSnippet]
    scores: list[NLIScore]
    verdict: str  # "Real" | "Misinformation" | "Disinformation"
    confidence: float
    probs: dict[str, float]
    reason: str


class AgentState(TypedDict, total=False):
    """End-to-end agent state.

    `total=False` means every field is optional — populated as nodes run.
    Required *input* fields (`claim`) are enforced by the API layer.
    """

    # --- Input ---
    claim: str
    image_bytes: bytes | None
    image_mime: str | None

    # --- Decomposer output ---
    sub_claims: list[str]

    # --- Retriever + NLI + sub-claim synthesizer output (per sub-claim) ---
    sub_results: list[SubClaimResult]

    # --- Final synthesizer output ---
    final_verdict: str
    final_confidence: float
    final_probs: dict[str, float]
    final_reason: str

    # --- Summarizer output (plain-English explanation, cites sources) ---
    summary: str
