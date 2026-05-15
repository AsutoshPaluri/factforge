"""Supabase client singleton.

Uses SUPABASE_SECRET_KEY (service_role) — bypasses RLS, has full read/write
access to all tables. NEVER expose this key to the frontend.

Initialised lazily on first call so module import is fast and tests that
don't touch the DB don't need real Supabase credentials.
"""

from __future__ import annotations

import structlog
from supabase import Client, create_client

from factforge.config import settings

logger = structlog.get_logger(__name__)


_instance: Client | None = None


def get_supabase() -> Client:
    """Return the process-wide Supabase client. Lazy-initialised."""
    global _instance
    if _instance is not None:
        return _instance

    url = settings.supabase_url
    key = settings.supabase_secret_key

    if not url or not key:
        raise RuntimeError(
            "Supabase not configured. Set SUPABASE_URL and SUPABASE_SECRET_KEY "
            "in .env (see .env.example)."
        )

    _instance = create_client(url, key)
    logger.info("supabase_client_initialised", url=url)
    return _instance


def is_configured() -> bool:
    """True if Supabase env vars are set. Safe to call without initialising."""
    return bool(settings.supabase_url and settings.supabase_secret_key)
