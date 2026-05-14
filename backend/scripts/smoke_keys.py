"""Smoke-test all configured external services.

Verifies that the keys in .env actually authenticate against Gemini,
Supabase, and Langfuse. Does NOT print any secret values.

Run from backend/:
    uv run python scripts/smoke_keys.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `factforge` importable when running as a standalone script
_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from factforge.config import settings  # noqa: E402


def check_gemini() -> str:
    """Round-trip a tiny request to verify the API key + model name."""
    if not settings.gemini_api_key:
        return "[FAIL] GEMINI_API_KEY not set in .env"
    try:
        from google import genai

        client = genai.Client(api_key=settings.gemini_api_key)
        resp = client.models.generate_content(
            model=settings.gemini_model,
            contents="Reply with exactly one word: pong",
        )
        text = (resp.text or "").strip().replace("\n", " ")
        return f"[ OK ] Gemini ({settings.gemini_model}) -> '{text[:50]}'"
    except Exception as e:
        return f"[FAIL] Gemini: {type(e).__name__}: {e}"


def check_supabase() -> str:
    """Hit the REST root with the secret key. 200 means auth + URL are correct."""
    if not settings.supabase_url or not settings.supabase_secret_key:
        return "[FAIL] Supabase URL or SECRET key not set in .env"
    try:
        import httpx

        r = httpx.get(
            f"{settings.supabase_url}/rest/v1/",
            headers={
                "apikey": settings.supabase_secret_key,
                "Authorization": f"Bearer {settings.supabase_secret_key}",
            },
            timeout=10.0,
        )
        if r.status_code == 200:
            return f"[ OK ] Supabase: REST endpoint reachable at {settings.supabase_url}"
        return f"[FAIL] Supabase: HTTP {r.status_code} - {r.text[:120]}"
    except Exception as e:
        return f"[FAIL] Supabase: {type(e).__name__}: {e}"


def check_langfuse() -> str:
    """Authenticate against the Langfuse cloud project."""
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return "[SKIP] Langfuse keys not set (optional)"
    try:
        from langfuse import Langfuse

        client = Langfuse(
            host=settings.langfuse_host,
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
        )
        ok = client.auth_check()
        if ok:
            return f"[ OK ] Langfuse: authenticated -> {settings.langfuse_host}"
        return "[FAIL] Langfuse: auth_check() returned False"
    except Exception as e:
        return f"[FAIL] Langfuse: {type(e).__name__}: {e}"


def main() -> None:
    print("=== factforge service smoke test ===\n")
    print(check_gemini())
    print(check_supabase())
    print(check_langfuse())
    print()


if __name__ == "__main__":
    main()
