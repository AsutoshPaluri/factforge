"""FastAPI entry point for factforge."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from factforge import __version__
from factforge.api.routes import router as api_router
from factforge.config import settings

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Pre-warm singletons so the first claim doesn't pay the model-load cost.

    On HuggingFace Spaces / Fly.io / etc, this means the container takes
    ~10-15s to become healthy, but the FIRST user request is fast.
    """
    # Lazy import so module-level `from factforge.main import app` (e.g. in
    # test_health.py) doesn't pull torch + transformers just to test routes.
    from factforge.clients.embeddings import get_embedder
    from factforge.clients.nli import get_verifier

    logger.info("lifespan_startup", version=__version__)
    verifier = await get_verifier()
    await verifier.warmup()
    embedder = get_embedder()
    await embedder.warmup()
    logger.info("lifespan_ready")

    yield

    logger.info("lifespan_shutdown")


app = FastAPI(
    title="factforge",
    version=__version__,
    description=(
        "Multimodal fact-checking agent with evidence-grounded NLI verification."
    ),
    lifespan=lifespan,
)


# CORS: allow the configured frontend + localhost during dev.
# For deployed prod, set FRONTEND_URL in the backend env to the Vercel URL.
_allowed_origins = [settings.frontend_url]
if "localhost" not in settings.frontend_url:
    _allowed_origins.append("http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    # Also accept any *.vercel.app preview deploy
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


app.include_router(api_router)


@app.get("/api/v1/health")
async def health() -> dict[str, str]:
    """Health check. Returns 200 if the service is up."""
    return {"status": "ok", "version": __version__}


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint — points to docs."""
    return {"service": "factforge", "version": __version__, "docs": "/docs"}
