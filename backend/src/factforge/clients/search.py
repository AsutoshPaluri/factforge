"""DuckDuckGo evidence retrieval (async).

Port of `misinfo_detection/utils/retrieval_classifier.py` retrieval layer,
adapted to async + httpx for FastAPI compatibility.

Pipeline:
    claim
        -> two DDG queries (raw + fact-check-biased)
        -> dedupe by URL/title
        -> drop non-English snippets
        -> optional: fetch each URL + extract paragraph text in parallel
        -> list[EvidenceSnippet]

No API keys. The `ddgs` library handles DDG's anti-bot rotation across
the html/lite/json backends.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

import httpx
import structlog
from ddgs import DDGS
from lxml import html as lhtml

from factforge.utils.text import content_words

logger = structlog.get_logger(__name__)


# --- Tunables ---
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
PAGE_FETCH_TIMEOUT_S = 5.0   # per-URL fetch deadline
PAGE_MAX_CHARS = 1800        # per-page budget for NLI (~512 DeBERTa tokens)
PAGE_MIN_CHARS = 200         # below this, fall back to DDG snippet
PAGE_FETCH_CONCURRENCY = 5   # max parallel page fetches

DDG_RETRIES = 3
DDG_BACKOFF_S = 1.5


# --- Public dataclass ---
@dataclass(frozen=True, slots=True)
class EvidenceSnippet:
    """One retrieved evidence snippet, ready for NLI scoring."""

    title: str
    snippet: str
    url: str
    source_is_fulltext: bool
    """True if `snippet` is extracted page-paragraph text; False if it's
    the DDG meta-description (page fetch failed or was too short)."""


# --- Language detection ---
# DeBERTa-v3-large MNLI-FEVER is English-only. Non-English text yields
# fake `contra ~ 0.99` scores that poison the verdict. We drop non-English
# snippets at retrieval to keep them out of the NLI pipeline entirely.
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
_ANY_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def _looks_english(
    text: str, min_chars: int = 60, min_latin_ratio: float = 0.85
) -> bool:
    """Heuristic: is `text` English (or close enough)?

    Rule: latin_letters / all_letters >= min_latin_ratio (default 0.85).

    "Earth is round"                                       -> ratio 1.0  -> True
    "Oficjalne centrum pomocy Google Earth, gdzie..."      -> ratio 0.65 -> False

    Short texts (< min_chars) trivially pass; downstream relevance gates
    catch them if they're irrelevant.
    """
    if not text or len(text) < min_chars:
        return True
    all_letters = _ANY_LETTER_RE.findall(text)
    if not all_letters:
        return True
    latin = _LATIN_LETTER_RE.findall(text)
    return (len(latin) / len(all_letters)) >= min_latin_ratio


# --- Boilerplate detection ---
# Strips publisher mission statements, subscribe nags, cookie banners.
_BOILERPLATE_PATTERNS = re.compile(
    r"\b("
    r"sign up for|subscribe to (?:our )?newsletter|join (?:our|us on|bezzy)|"
    r"this content includes information|our team of editors|"
    r"committed to bringing you|expert-driven content|researched,? expert|"
    r"fact-checked to ensure|editorial process|editorial team|"
    r"can't get enough\?|connect with us|all rights reserved|"
    r"privacy policy|cookie policy|accept all cookies|"
    r"share this article|related articles?:|read more:|see also:|"
    r"originally published|last updated on|advertisement"
    r")\b",
    re.IGNORECASE,
)


def _is_boilerplate(text: str) -> bool:
    return bool(_BOILERPLATE_PATTERNS.search(text))


# --- Page fetching ---
async def fetch_page_text(
    client: httpx.AsyncClient,
    url: str,
    claim: str | None = None,
) -> str:
    """Fetch URL, strip chrome, return claim-relevant paragraph text.

    Pipeline:
      1. GET with browser UA, follow redirects
      2. Strip <script>, <style>, <nav>, <footer>, <aside>, <header>, <form>
      3. Keep <p> with len > 40 chars, drop boilerplate
      4. If `claim` given, sort paragraphs by claim-word overlap (most
         relevant survives the PAGE_MAX_CHARS truncation)
      5. Return up to PAGE_MAX_CHARS

    Returns "" on any failure (blocked, 404, non-HTML, parse error).
    """
    try:
        r = await client.get(url)
    except Exception:
        return ""

    if r.status_code != 200:
        return ""
    if "html" not in r.headers.get("content-type", "").lower():
        return ""

    try:
        doc = lhtml.fromstring(r.text)
    except Exception:
        return ""

    for el in doc.xpath(
        "//script | //style | //noscript | //nav | "
        "//footer | //aside | //header | //form"
    ):
        parent = el.getparent()
        if parent is not None:
            parent.remove(el)

    paragraphs = [p.text_content().strip() for p in doc.xpath("//p")]
    paragraphs = [p for p in paragraphs if len(p) > 40 and not _is_boilerplate(p)]
    if not paragraphs:
        return ""

    if claim:
        c_words = content_words(claim)

        def _score(p: str) -> int:
            return len({w.lower() for w in re.findall(r"\b[A-Za-z]+\b", p)} & c_words)

        paragraphs.sort(key=_score, reverse=True)

    text = re.sub(r"\s+", " ", " ".join(paragraphs)).strip()
    return text[:PAGE_MAX_CHARS]


# --- DDG search ---
async def ddg_search(query: str, k: int = 5) -> list[dict[str, str]]:
    """DuckDuckGo search via the sync `ddgs` library, wrapped in `to_thread`.

    The library rotates between DDG backends (html, lite, json) so single-
    endpoint rate-limits don't kill the call. We add retries for transient
    network errors.

    verify=False: `ddgs` randomly attempts TLS 1.3, which macOS LibreSSL
    doesn't support. DDG results are public, so skipping TLS verify is fine.
    """

    def _sync_search() -> list[dict]:
        return list(DDGS(verify=False).text(query, max_results=k))

    last_err: Exception | None = None
    for attempt in range(DDG_RETRIES):
        try:
            raw = await asyncio.to_thread(_sync_search)
            return [
                {
                    "title": (r.get("title") or "").strip(),
                    "snippet": (r.get("body") or "").strip(),
                    "url": (r.get("href") or "").strip(),
                }
                for r in raw
                if (r.get("title") or "").strip() and (r.get("body") or "").strip()
            ]
        except Exception as e:
            last_err = e
            if attempt < DDG_RETRIES - 1:
                await asyncio.sleep(DDG_BACKOFF_S * (attempt + 1))

    logger.warning("ddg_search_failed", query=query, error=str(last_err))
    return []


def _build_fact_check_query(claim: str) -> str:
    """Bias DDG toward fact-checking / explainer sites.

    A naive DDG search for a claim often returns sites that *repeat* the
    claim (myth phrases echo across content farms). Appending 'fact check
    OR myth OR true or false' nudges DDG toward explainer / debunking sites.
    """
    return f"{claim} fact check OR myth OR true or false"


# --- Main entry point ---
async def retrieve_evidence(
    claim: str,
    k: int = 5,
    enrich: bool = True,
) -> list[EvidenceSnippet]:
    """End-to-end retrieval: dual-query DDG -> dedupe -> optional page enrichment.

    Args:
        claim: the user's claim
        k: max results per DDG query (we fire two, so up to 2k merged before dedupe)
        enrich: if True, fetch each URL in parallel and replace the DDG meta-
            description with extracted paragraph text. Disable for fast
            preview / smoke tests.

    Returns:
        Deduped list of EvidenceSnippet. Empty list on total search failure.
    """
    queries = [claim, _build_fact_check_query(claim)]

    # Fire both queries in parallel
    search_results = await asyncio.gather(*(ddg_search(q, k=k) for q in queries))

    # Merge + dedupe (URL then title fallback) + drop non-English
    seen: set[str] = set()
    merged: list[dict[str, str]] = []
    for results in search_results:
        for r in results:
            key = (r.get("url") or r.get("title") or "").lower()
            if not key or key in seen:
                continue
            seen.add(key)
            snip = r["snippet"]
            if not snip or not _looks_english(snip):
                continue
            merged.append(r)

    if not merged:
        logger.info("retrieve_evidence_empty", claim=claim)
        return []

    if not enrich:
        return [
            EvidenceSnippet(
                title=r["title"],
                snippet=r["snippet"],
                url=r.get("url", ""),
                source_is_fulltext=False,
            )
            for r in merged
        ]

    # Parallel page-text enrichment, bounded by semaphore
    sem = asyncio.Semaphore(PAGE_FETCH_CONCURRENCY)

    async with httpx.AsyncClient(
        headers={
            "User-Agent": BROWSER_UA,
            "Accept-Language": "en-US,en;q=0.9",
        },
        follow_redirects=True,
        timeout=PAGE_FETCH_TIMEOUT_S,
    ) as client:

        async def _fetch_one(url: str) -> str:
            async with sem:
                return await fetch_page_text(client, url, claim=claim)

        page_texts = await asyncio.gather(
            *(_fetch_one(r["url"]) for r in merged),
            return_exceptions=True,
        )

    enriched: list[EvidenceSnippet] = []
    for r, page_text in zip(merged, page_texts, strict=True):
        page_text_str = "" if isinstance(page_text, BaseException) else page_text
        if (
            len(page_text_str) >= PAGE_MIN_CHARS
            and _looks_english(page_text_str)
        ):
            snippet, is_full = page_text_str, True
        else:
            snippet, is_full = r["snippet"], False

        enriched.append(
            EvidenceSnippet(
                title=r["title"],
                snippet=snippet,
                url=r.get("url", ""),
                source_is_fulltext=is_full,
            )
        )

    return enriched
