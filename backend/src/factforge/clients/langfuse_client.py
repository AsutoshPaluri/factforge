"""Langfuse observability — LangGraph callback handler singleton.

Wraps Langfuse's LangChain-style CallbackHandler. Since LangGraph uses
LangChain's callback machinery internally, attaching this handler to
graph.ainvoke() auto-instruments every node:
  - decomposer, retriever, nli, synthesizer, summarizer
  - inputs / outputs / latencies / errors at each step
  - one trace per /claims request

Graceful by design:
  - If LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY aren't set: return None.
  - If keys are set but auth fails: log a warning, return None.
  - Either way, the agent pipeline still runs normally — observability is
    a nice-to-have, never a hard dependency.

Usage:
    handler = get_callback_handler()
    config = {"callbacks": [handler]} if handler else {}
    final_state = await graph.ainvoke(state, config=config)
"""

from __future__ import annotations

from typing import Any

import structlog

from factforge.config import settings

logger = structlog.get_logger(__name__)


_handler: Any | None = None
_handler_checked: bool = False


def is_configured() -> bool:
    """True if Langfuse env vars are present. Safe to call without init."""
    return bool(
        settings.langfuse_public_key
        and settings.langfuse_secret_key
        and settings.langfuse_host
    )


def get_callback_handler() -> Any | None:
    """Return the Langfuse CallbackHandler singleton, or None if unavailable.

    First call attempts to construct + auth-check the handler. If auth
    succeeds, the handler is cached and reused for every invocation.
    If auth fails (or keys are missing), returns None and never tries
    again — the agent runs normally without tracing.
    """
    global _handler, _handler_checked

    if _handler_checked:
        return _handler

    _handler_checked = True

    if not is_configured():
        logger.info("langfuse_skipped_not_configured")
        return None

    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler  # v3+ path

        # Probe auth first so we don't ship a broken handler downstream.
        client = Langfuse(
            host=settings.langfuse_host,
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
        )
        if not client.auth_check():
            logger.warning(
                "langfuse_auth_check_failed",
                host=settings.langfuse_host,
                hint=(
                    "Regenerate keys at the Langfuse project's Settings -> "
                    "API Keys, paste into .env, and restart the backend. "
                    "Verify LANGFUSE_HOST matches the region of the project."
                ),
            )
            return None

        # In Langfuse v3+, the CallbackHandler reads credentials from the
        # already-initialised global Langfuse client — we just pass it
        # through here (the client was constructed above with our creds).
        _handler = CallbackHandler()
        logger.info("langfuse_initialised", host=settings.langfuse_host)
        return _handler

    except Exception as e:
        # Any failure (network, import, version mismatch, etc.) -> no tracing,
        # agent keeps running.
        logger.warning("langfuse_init_failed", error=str(e))
        return None


def make_config(extra_metadata: dict | None = None) -> dict:
    """Build a LangGraph invoke config that includes the Langfuse callback
    if available.

    `extra_metadata` is attached to the trace for filtering in the dashboard
    (e.g. {"endpoint": "POST /claims/refine", "feedback": "..."}).
    """
    config: dict = {}
    handler = get_callback_handler()
    if handler is not None:
        config["callbacks"] = [handler]
    if extra_metadata:
        # LangGraph runs accept a `metadata` block in config that Langfuse
        # surfaces as searchable tags on the trace.
        config["metadata"] = extra_metadata
    return config
