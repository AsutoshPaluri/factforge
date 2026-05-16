# =========================================================================
# factforge backend — Dockerfile for HuggingFace Spaces (Docker template)
#
# HF Spaces free tier: 2 vCPU, 16GB RAM, no GPU. Fits our 1.6GB DeBERTa
# model comfortably. Listens on $PORT (default 7860 on HF).
#
# IMPORTANT: HF Spaces runs the container as uid 1000 (the `user` user).
# All filesystem state — especially the venv — must be owned by that user
# or `uv run` at startup hits Permission denied when it tries to update
# the venv's INSTALLER metadata.
#
# Build:  docker build -t factforge .
# Run:    docker run --rm -p 7860:7860 --env-file .env factforge
# =========================================================================

FROM python:3.12-slim

# --- System deps (as root) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libxml2 \
        libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# Create the HF-conventional uid-1000 user
RUN useradd -m -u 1000 user

# Install uv system-wide so the user can use it (no per-user install needed)
RUN pip install --no-cache-dir uv==0.11.14

# Make /app and hand ownership to user BEFORE switching users — that way
# the venv we create below ends up entirely user-owned.
RUN mkdir -p /app && chown -R user:user /app

WORKDIR /app
USER user

ENV PATH="/app/.venv/bin:/home/user/.local/bin:${PATH}" \
    HF_HOME="/home/user/.cache/huggingface" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# --- Python deps (as user, so .venv is user-owned) ---
COPY --chown=user:user pyproject.toml uv.lock* ./

# First pass: install deps without the project itself (better layer caching)
RUN uv sync --frozen --no-dev --no-install-project 2>/dev/null \
    || uv sync --no-dev --no-install-project

# Now copy source + install the project
COPY --chown=user:user src ./src

RUN uv sync --no-dev --frozen 2>/dev/null \
    || uv sync --no-dev

# --- Pre-download models at build time ---
# Uses the venv's python directly (not `uv run`) to avoid any sync side-effects.
# Falls back gracefully if the network is restricted during build — models
# will then download at first inference instead.
RUN /app/.venv/bin/python -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel; \
nli='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'; \
emb='sentence-transformers/all-MiniLM-L6-v2'; \
AutoTokenizer.from_pretrained(nli); \
AutoModelForSequenceClassification.from_pretrained(nli); \
AutoTokenizer.from_pretrained(emb); \
AutoModel.from_pretrained(emb); \
print('models pre-cached')" \
    || echo "[warn] model pre-download skipped (will download at runtime)"

EXPOSE 7860

# Direct path to the venv's uvicorn — skip `uv run` so there's no sync
# attempt at container startup (that's what caused the Permission-denied
# at runtime previously).
CMD ["sh", "-c", "/app/.venv/bin/uvicorn factforge.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
