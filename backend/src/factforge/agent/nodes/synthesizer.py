"""Node 4: Verdict synthesizer.

Per-sub-claim aggregation logic — ported from
misinfo_detection/utils/retrieval_classifier.py (classify() method).

Steps for each sub-claim:
  1. Topical-relevance gate — drop off-topic evidence before aggregation
  2. Per-snippet weighting:
        weight = 0.5 + 0.5 * word_overlap + 0.3 * numeric_overlap
        info_score = (entailment + contradiction) * weight
     Title-aware contradiction penalty: if contra > 0.5 but the source
     title barely matches the sub-claim, scale info_score by 0.25.
  3. Max + mean (top-3) blend, normalised to probabilities.
  4. Hybrid binary-with-abstain verdict:
        - Default: pick between Credible and Not Credible (binary).
        - Fall back to Uncertain ONLY when:
            * p_uncertain (neutral) > both binary probs, OR
            * binary leader has p < 0.40 AND gap to the other is < 0.10
              (genuinely 50/50 evidence — better to abstain than guess)

Final-verdict combination across sub-claims:
  - Any Not Credible -> Not Credible
  - Else any Uncertain -> Uncertain
  - Else Credible
  Confidence = mean of per-sub-claim confidences.
"""

from __future__ import annotations

import structlog

from factforge.agent.state import AgentState, SubClaimResult
from factforge.utils.text import (
    claim_overlap,
    content_words,
    numeric_overlap,
    words,
)

logger = structlog.get_logger(__name__)


# When evidence is insufficient or all snippets fail the relevance gate.
_FALLBACK_VERDICT = "Uncertain"
_FALLBACK_PROBS = {"Credible": 0.33, "Uncertain": 0.34, "Not Credible": 0.33}

# Thresholds for routing to Uncertain (the abstain bucket). Tuned to be
# conservative: we only abstain when evidence is genuinely indecisive.
_BINARY_COMMIT_FLOOR = 0.40       # binary leader must clear this prob
_BINARY_TIE_GAP = 0.10            # leader-vs-other gap below this = tied


def _resolve_verdict(
    p_credible: float, p_uncertain: float, p_not_credible: float
) -> str:
    """Pick Credible / Not Credible / Uncertain from normalised probs.

    Prefers binary verdicts. Only abstains to Uncertain when:
      - neutral evidence dominates (p_uncertain is the max), OR
      - the two binary probs are essentially tied AND neither is strong.
    """
    binary_max = max(p_credible, p_not_credible)
    binary_min = min(p_credible, p_not_credible)

    # Case 1: neutral evidence dominates -> truly insufficient signal
    if p_uncertain > binary_max:
        return "Uncertain"

    # Case 2: binary leader is weak AND essentially tied with the other -> abstain
    if binary_max < _BINARY_COMMIT_FLOOR and (binary_max - binary_min) < _BINARY_TIE_GAP:
        return "Uncertain"

    return "Credible" if p_credible > p_not_credible else "Not Credible"


def _aggregate_sub_claim(sr: SubClaimResult) -> SubClaimResult:
    """Compute verdict + confidence + probs for one sub-claim from its NLI scores."""
    sub_claim = sr["sub_claim"]
    evidence = sr["evidence"]
    scores = sr["scores"]

    if not evidence or not scores or len(evidence) != len(scores):
        return {
            **sr,
            "verdict": _FALLBACK_VERDICT,
            "confidence": 0.34,
            "probs": dict(_FALLBACK_PROBS),
            "reason": "no evidence retrieved or NLI did not score it",
        }

    # Topical-relevance gate: need >=2 shared content words (or 1 if the
    # claim itself has only 1-2 content words). Numeric overlap is also a
    # rescue path for "100 GHz" / "10%" style claims.
    sub_words = content_words(sub_claim)
    min_overlap = 1 if len(sub_words) <= 2 else 2

    scored: list[tuple[int, float]] = []  # (evidence_idx, info_score)
    for i, (ev, sc) in enumerate(zip(evidence, scores, strict=True)):
        sent_words = words(ev.snippet)
        shared = sent_words & sub_words
        num_ov = numeric_overlap(ev.snippet, sub_claim)

        if len(shared) < min_overlap and num_ov == 0:
            continue

        word_ov = claim_overlap(ev.snippet, sub_claim)
        title_w = words(ev.title)
        title_ov = (
            len(title_w & sub_words) / max(len(sub_words), 1) if sub_words else 0.0
        )

        weight = 0.5 + 0.5 * word_ov + 0.3 * num_ov
        info = (sc.entailment + sc.contradiction) * weight

        # Title-aware contradiction penalty: a contradicting snippet from a
        # source whose title barely overlaps the claim is usually NLI
        # misreading (off-topic), not real contradiction.
        if sc.contradiction > 0.5 and title_ov < 0.15:
            info *= 0.25

        scored.append((i, info))

    if not scored:
        return {
            **sr,
            "verdict": _FALLBACK_VERDICT,
            "confidence": 0.34,
            "probs": dict(_FALLBACK_PROBS),
            "reason": (
                f"all {len(evidence)} retrieved snippets failed the "
                f"topical-relevance gate (need >={min_overlap} shared "
                f"content words with the sub-claim)"
            ),
        }

    # Max + top-3-mean blend (deterministic sort by info_score then idx)
    surviving_scores = [scores[i] for (i, _) in scored]
    entail_max = max(s.entailment for s in surviving_scores)
    contra_max = max(s.contradiction for s in surviving_scores)

    scored.sort(key=lambda t: (-t[1], t[0]))  # info_score desc, idx asc
    top_idx = [i for (i, _) in scored[:3]]
    top = [scores[i] for i in top_idx]

    mean_e = sum(s.entailment for s in top) / len(top)
    mean_n = sum(s.neutral for s in top) / len(top)
    mean_c = sum(s.contradiction for s in top) / len(top)

    agg_e = 0.5 * entail_max + 0.5 * mean_e
    agg_c = 0.5 * contra_max + 0.5 * mean_c
    agg_n = mean_n

    total = agg_e + agg_c + agg_n
    if total <= 0:
        agg_e, agg_c, agg_n, total = 1 / 3, 1 / 3, 1 / 3, 1.0

    p_credible = agg_e / total
    p_not_credible = agg_c / total
    p_uncertain = agg_n / total

    probs = {
        "Credible": round(p_credible, 4),
        "Uncertain": round(p_uncertain, 4),
        "Not Credible": round(p_not_credible, 4),
    }
    verdict = _resolve_verdict(p_credible, p_uncertain, p_not_credible)
    confidence = probs[verdict]

    return {
        **sr,
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "probs": probs,
        "reason": (
            f"aggregated {len(scored)} relevant snippets "
            f"(out of {len(evidence)} retrieved)"
        ),
    }


def _combine_sub_verdicts(sub_results: list[SubClaimResult]) -> dict:
    """Combine per-sub-claim verdicts into a final claim verdict.

    Worst-of policy: any Not Credible -> Not Credible; else any Uncertain
    -> Uncertain; else Credible. Confidence = mean of sub-claim confidences.
    """
    if not sub_results:
        return {
            "final_verdict": _FALLBACK_VERDICT,
            "final_confidence": 0.34,
            "final_probs": dict(_FALLBACK_PROBS),
            "final_reason": "no sub-claims to aggregate",
        }

    verdicts = [sr["verdict"] for sr in sub_results]
    confidences = [sr["confidence"] for sr in sub_results]

    if "Not Credible" in verdicts:
        final = "Not Credible"
    elif "Uncertain" in verdicts:
        final = "Uncertain"
    else:
        final = "Credible"

    # Final probs: per-class, the MAX across sub-claims for that class
    # (so a single high-not-credible sub-claim carries through).
    final_probs = {
        "Credible": round(
            min((sr["probs"].get("Credible", 0.0) for sr in sub_results)), 4
        ),
        "Uncertain": round(
            max((sr["probs"].get("Uncertain", 0.0) for sr in sub_results)), 4
        ),
        "Not Credible": round(
            max((sr["probs"].get("Not Credible", 0.0) for sr in sub_results)), 4
        ),
    }

    return {
        "final_verdict": final,
        "final_confidence": round(sum(confidences) / len(confidences), 4),
        "final_probs": final_probs,
        "final_reason": (
            f"aggregated {len(sub_results)} sub-claim verdict(s): "
            + ", ".join(f"{sr['sub_claim'][:40]!r}={sr['verdict']}" for sr in sub_results)
        ),
    }


async def synthesizer_node(state: AgentState) -> dict:
    """Compute per-sub-claim verdicts + combine into final verdict."""
    sub_results = state.get("sub_results") or []
    aggregated = [_aggregate_sub_claim(sr) for sr in sub_results]
    final = _combine_sub_verdicts(aggregated)

    logger.info(
        "synthesizer_done",
        final_verdict=final["final_verdict"],
        final_confidence=final["final_confidence"],
        n_sub_claims=len(aggregated),
    )

    return {"sub_results": aggregated, **final}
