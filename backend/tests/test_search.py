"""Smoke tests for DDG evidence retrieval — hits the live DuckDuckGo endpoint.

These tests skip rather than fail when DDG returns 0 results (a transient
condition we can't control). The structure of returned data is still
validated when results are present.
"""

from __future__ import annotations

import pytest

from factforge.clients.search import (
    _looks_english,
    ddg_search,
    retrieve_evidence,
)


@pytest.mark.asyncio
async def test_ddg_search_returns_dicts() -> None:
    results = await ddg_search("Python programming language", k=3)
    if not results:
        pytest.skip("DDG returned 0 results (likely transient)")
    for r in results:
        assert "title" in r and r["title"]
        assert "snippet" in r and r["snippet"]
        assert "url" in r


@pytest.mark.asyncio
async def test_retrieve_evidence_no_enrich() -> None:
    """Fast path: no page fetching. Should return DDG snippets directly."""
    snippets = await retrieve_evidence(
        "vaccines cause autism", k=3, enrich=False
    )
    if not snippets:
        pytest.skip("DDG returned 0 results (likely transient)")
    assert len(snippets) > 0
    for s in snippets:
        assert s.title
        assert s.snippet
        assert s.url.startswith("http")
        assert s.source_is_fulltext is False


def test_looks_english_basic() -> None:
    assert _looks_english("The earth is round and orbits the sun.")
    assert _looks_english("")  # empty trivially passes
    assert _looks_english("hi")  # too short — heuristic abstains

    # Cyrillic — 0% Latin letters, easily caught by the 0.85 threshold.
    # This is the actual failure mode we care about: NLI returns garbage
    # for non-English-script text, so we drop it pre-NLI.
    russian = (
        "Привет, как дела? Сегодня прекрасный день для прогулки в парке. "
        "Я надеюсь, что мы скоро увидимся и поговорим о наших планах."
    )
    assert not _looks_english(russian)


def test_looks_english_threshold() -> None:
    # Mostly English with a few diacritics — should still pass (Polish-style
    # mild diacritics are tolerated; DeBERTa handles them OK).
    mixed = (
        "The café was full of students writing essays about climate change. "
        "Outside, the weather was unusually warm for November."
    )
    assert _looks_english(mixed)
