"""NLI smoke tests — actually load the 1.6GB DeBERTa-MNLI-FEVER model and infer.

Marked @pytest.mark.slow because:
  - First run downloads ~1.6GB from HuggingFace (one-time, cached after)
  - Each test does a CPU forward pass (~2-5s on M-series)

Skip with: pytest -m "not slow"
Run only these: pytest -m slow
"""

from __future__ import annotations

import pytest

from factforge.clients.nli import NLIScore, get_verifier


@pytest.mark.slow
@pytest.mark.asyncio
async def test_nli_clear_entailment() -> None:
    """A premise that clearly supports the hypothesis should entail."""
    verifier = await get_verifier()
    score = await verifier.score_one(
        premise="Earth is approximately spherical, slightly oblate at the poles.",
        hypothesis="The earth is round.",
    )
    assert isinstance(score, NLIScore)
    assert score.entailment > 0.5
    assert score.contradiction < 0.2


@pytest.mark.slow
@pytest.mark.asyncio
async def test_nli_clear_contradiction() -> None:
    """A premise that clearly refutes the hypothesis should contradict."""
    verifier = await get_verifier()
    score = await verifier.score_one(
        premise="Earth is approximately spherical, slightly oblate at the poles.",
        hypothesis="The earth is flat.",
    )
    assert score.contradiction > 0.5
    assert score.entailment < 0.2


@pytest.mark.slow
@pytest.mark.asyncio
async def test_nli_batch_order_preserved() -> None:
    """Batch results come back in the same order as input premises."""
    verifier = await get_verifier()
    scores = await verifier.score(
        premises=[
            "Earth is approximately spherical.",  # entails 'round'
            "Earth is flat.",                     # contradicts 'round'
            "Paris is the capital of France.",    # unrelated -> neutral
        ],
        hypothesis="The earth is round.",
    )
    assert len(scores) == 3
    # Spherical should entail 'round' more strongly than flat does
    assert scores[0].entailment > scores[1].entailment
    # Unrelated should lean neutral
    assert scores[2].neutral > scores[2].entailment
    assert scores[2].neutral > scores[2].contradiction


@pytest.mark.slow
@pytest.mark.asyncio
async def test_nli_probabilities_sum_to_one() -> None:
    """The three classes are softmax outputs and should sum to ~1.0."""
    verifier = await get_verifier()
    score = await verifier.score_one(
        premise="The sky is blue on a clear day.",
        hypothesis="The grass is green.",
    )
    total = score.entailment + score.neutral + score.contradiction
    assert 0.99 < total < 1.01


@pytest.mark.asyncio
async def test_nli_empty_premises_returns_empty() -> None:
    """No premises -> no scores. Doesn't load the model."""
    verifier = await get_verifier()
    scores = await verifier.score(premises=[], hypothesis="anything")
    assert scores == []
