"""Shared text utilities for claims and evidence snippets.

These helpers are used both by the retrieval layer (filtering snippets
by topical relevance) and by the verdict synthesizer (weighting NLI
scores by claim-overlap).
"""

from __future__ import annotations

import re

# Common English stop words — removed before computing claim/snippet overlap
# so the gate doesn't consider "the", "is", "of" as shared content.
STOP_WORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "of", "in", "on", "at", "for", "with", "to", "from", "by", "as",
        "that", "this", "these", "those", "it", "its", "has", "have", "had",
        "do", "does", "did", "and", "or", "but", "not", "no", "yes", "if",
        "than", "then", "i", "you", "we", "they", "he", "she", "his", "her",
        "their", "my", "our", "can", "could", "would", "should", "may",
        "might", "must", "shall", "will",
    }
)

_WORD_RE = re.compile(r"\b[A-Za-z]+\b")
_NUM_RE = re.compile(r"\d+")


def words(text: str) -> set[str]:
    """Lowercased set of ASCII word tokens."""
    return {w.lower() for w in _WORD_RE.findall(text)}


def content_words(text: str) -> set[str]:
    """Lowercased word set with stop words removed."""
    return words(text) - STOP_WORDS


def claim_overlap(sentence: str, claim: str) -> float:
    """Fraction of claim content-words appearing in the sentence (0.0 - 1.0)."""
    c_words = content_words(claim)
    if not c_words:
        return 0.0
    s_words = words(sentence)
    return len(s_words & c_words) / len(c_words)


def numeric_overlap(sentence: str, claim: str) -> float:
    """Fraction of claim-numbers that appear in the sentence (0.0 - 1.0).

    Matters for claims like '5G operates at 60 GHz' or 'humans use only 10%
    of their brain' — the number is the whole point, so a snippet that
    contains it is strong evidence even with low word overlap.
    """
    c_nums = set(_NUM_RE.findall(claim))
    if not c_nums:
        return 0.0
    s_nums = set(_NUM_RE.findall(sentence))
    return len(c_nums & s_nums) / len(c_nums)


# --- Perception-verb normalisation ---
# Collapses "X looks/seems/appears Y" -> "X is Y" so NLI handles both
# phrasings identically. Without this, the same claim gets dramatically
# different verdicts depending on phrasing:
#     "The Earth is flat"                -> Disinformation 0.85
#     "Earth looks flat from space"      -> Disinformation 0.54  (weaker)
# because NLI reads "Earth is a spheroid" as not-quite-contradicting
# "Earth looks flat" (perception != ontology). Retrieval still uses the
# user's original wording (DDG works better on natural-language queries);
# only the NLI hypothesis gets normalised.

_PERCEPTION_AT_START_RE = re.compile(
    r"^\s*(?:it\s+)?(?:looks?|seems?|appears?|appeared|appearing|seeming)\s+"
    r"(?:like|as\s+(?:if|though)|that)\s+",
    re.IGNORECASE,
)
_PERCEPTION_VERB_RE = re.compile(
    r"\b(looks?|seems?|appears?|appeared|appearing|seeming)\b",
    re.IGNORECASE,
)


def normalise_perception_verbs(claim: str) -> str:
    """Strip perception framing so NLI handles 'X looks Y' the same as 'X is Y'.

    Examples:
        "Earth looks flat when viewed from space"
            -> "Earth is flat when viewed from space"
        "It seems that vaccines cause autism"
            -> "vaccines cause autism"
        "The cake looks delicious"
            -> "The cake is delicious"
    """
    if not claim:
        return claim
    out = _PERCEPTION_AT_START_RE.sub("", claim).strip()
    out = _PERCEPTION_VERB_RE.sub("is", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out or claim  # never return empty
