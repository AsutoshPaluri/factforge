"""End-to-end agent smoke tests.

Hits the FULL pipeline against live services:
    Gemini (decomposer) -> DuckDuckGo (retriever) ->
    DeBERTa-MNLI-FEVER (NLI) -> synthesizer

Marked @pytest.mark.slow because the first run downloads the 1.6GB NLI
model (cached after) and the full pipeline takes ~10-30s per claim.
"""

from __future__ import annotations

import pytest

from factforge.agent.graph import get_graph
from factforge.config import settings


def _has_gemini_key() -> bool:
    return bool(settings.gemini_api_key)


@pytest.mark.slow
@pytest.mark.skipif(not _has_gemini_key(), reason="GEMINI_API_KEY not set")
@pytest.mark.asyncio
async def test_e2e_obvious_disinformation() -> None:
    """A claim that's clearly debunked online should get classified as
    Disinformation (or at minimum, not Real)."""
    graph = get_graph()
    final_state = await graph.ainvoke({"claim": "The earth is flat"})

    assert "final_verdict" in final_state
    assert final_state["final_verdict"] in {
        "Real",
        "Misinformation",
        "Disinformation",
    }
    assert 0.0 <= final_state["final_confidence"] <= 1.0
    assert set(final_state["final_probs"].keys()) == {
        "Disinformation",
        "Misinformation",
        "Real",
    }

    # We expect a clearly-debunked claim to lean away from Real
    assert final_state["final_verdict"] != "Real", (
        f"expected non-Real for 'earth is flat', got {final_state['final_verdict']} "
        f"with probs {final_state['final_probs']}"
    )


@pytest.mark.slow
@pytest.mark.skipif(not _has_gemini_key(), reason="GEMINI_API_KEY not set")
@pytest.mark.asyncio
async def test_e2e_compound_claim_decomposes() -> None:
    """A compound claim should be decomposed into multiple sub-claims."""
    graph = get_graph()
    final_state = await graph.ainvoke(
        {
            "claim": (
                "Vaccines cause autism and humans only use 10 percent of "
                "their brains."
            )
        }
    )

    sub_claims = final_state.get("sub_claims") or []
    assert len(sub_claims) >= 2, (
        f"compound claim should decompose into 2+ sub-claims, "
        f"got {len(sub_claims)}: {sub_claims}"
    )

    sub_results = final_state.get("sub_results") or []
    assert len(sub_results) == len(sub_claims)
    for sr in sub_results:
        assert sr["verdict"] in {"Real", "Misinformation", "Disinformation"}
