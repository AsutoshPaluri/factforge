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
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

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


# --- Domain quality filters ---
# Blocked: known-poor sources for fact-checking — tabloids, listicles,
# Q&A forums, social-media snippets (just video descriptions). These hurt
# NLI verdict accuracy disproportionately when they leak into a small
# evidence pool.
BLOCKED_DOMAINS = frozenset({
    # Tabloids
    "metro.co.uk", "dailymail.co.uk", "mirror.co.uk",
    "the-sun.com", "nypost.com", "thesun.co.uk",
    # Listicle / clickbait
    "buzzfeed.com", "cracked.com", "boredpanda.com",
    # Q&A and forums
    "quora.com", "reddit.com", "answers.com", "answers.yahoo.com",
    # Social-media snippets are just video/post descriptions — useless for NLI
    "youtube.com", "youtu.be", "tiktok.com", "twitter.com", "x.com",
    "facebook.com", "instagram.com", "pinterest.com",
    # AI-generated / aggregator junk
    "actforlibraries.org",  # SEO-spam-ish
})


def _normalised_domain(url: str) -> str:
    """Lowercased hostname with leading 'www.' stripped. '' on parse failure."""
    try:
        host = urlparse(url).hostname or ""
        return host.lower().removeprefix("www.")
    except Exception:
        return ""


def _is_blocked_domain(url: str) -> bool:
    """True if the URL's domain (or any parent domain) is in BLOCKED_DOMAINS."""
    d = _normalised_domain(url)
    if not d:
        return False
    if d in BLOCKED_DOMAINS:
        return True
    # Match suffixes too: "uk.reddit.com" -> blocked because reddit.com is blocked
    for bd in BLOCKED_DOMAINS:
        if d.endswith("." + bd):
            return True
    return False


# --- Evidence cache ---
# Deterministic-by-claim: same claim text -> same retrieved evidence ->
# same NLI scores -> same verdict. This is what fixes "I ran it twice and
# got different answers." 7-day TTL because web pages drift.
_CACHE_DIR = Path.home() / ".cache" / "factforge" / "evidence"
_CACHE_TTL_S = 7 * 24 * 3600  # 7 days


def _cache_key(claim: str, k: int, enrich: bool) -> str:
    """Stable hash of (normalised claim, k, enrich) for cache lookup."""
    normalised = claim.strip().lower()
    raw = f"{normalised}|k={k}|enrich={int(enrich)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _cache_path(key: str) -> Path:
    return _CACHE_DIR / f"{key}.json"


def _cache_get(claim: str, k: int, enrich: bool) -> list[EvidenceSnippet] | None:
    """Read cached evidence if present and within TTL. None if miss/expired."""
    path = _cache_path(_cache_key(claim, k, enrich))
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        if time.time() - raw.get("ts", 0) > _CACHE_TTL_S:
            return None
        return [EvidenceSnippet(**item) for item in raw["evidence"]]
    except Exception as e:
        logger.warning("evidence_cache_read_failed", error=str(e))
        return None


def _cache_put(
    claim: str, k: int, enrich: bool, evidence: list[EvidenceSnippet]
) -> None:
    """Persist evidence list under a stable claim-derived key."""
    if not evidence:
        return  # never cache empties — retrying might succeed next time
    path = _cache_path(_cache_key(claim, k, enrich))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": time.time(),
            "claim": claim.strip(),
            "k": k,
            "enrich": enrich,
            "evidence": [asdict(ev) for ev in evidence],
        }
        path.write_text(json.dumps(payload))
    except Exception as e:
        logger.warning("evidence_cache_write_failed", error=str(e))


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
    # --- Cache hit short-circuit: same claim -> same evidence -> same verdict ---
    cached = _cache_get(claim, k, enrich)
    if cached is not None:
        logger.info(
            "retrieve_evidence_cache_hit", claim=claim[:80], n=len(cached)
        )
        return cached

    queries = [claim, _build_fact_check_query(claim)]

    # Fire both queries in parallel
    search_results = await asyncio.gather(*(ddg_search(q, k=k) for q in queries))

    # Merge + dedupe + drop non-English + drop blocked domains
    seen: set[str] = set()
    merged: list[dict[str, str]] = []
    n_blocked = 0
    for results in search_results:
        for r in results:
            key = (r.get("url") or r.get("title") or "").lower()
            if not key or key in seen:
                continue
            seen.add(key)
            url = r.get("url", "")
            if url and _is_blocked_domain(url):
                n_blocked += 1
                continue
            snip = r["snippet"]
            if not snip or not _looks_english(snip):
                continue
            merged.append(r)

    if n_blocked:
        logger.info(
            "retrieve_evidence_blocked_filtered",
            claim=claim[:80],
            n_blocked=n_blocked,
        )

    if not merged:
        logger.info("retrieve_evidence_empty", claim=claim)
        return []

    if not enrich:
        result_no_enrich = [
            EvidenceSnippet(
                title=r["title"],
                snippet=r["snippet"],
                url=r.get("url", ""),
                source_is_fulltext=False,
            )
            for r in merged
        ]
        _cache_put(claim, k, enrich, result_no_enrich)
        return result_no_enrich

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

    _cache_put(claim, k, enrich, enriched)
    return enriched
