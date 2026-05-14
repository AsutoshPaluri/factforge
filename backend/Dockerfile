# =========================================================================
# factforge backend — Dockerfile for HuggingFace Spaces (Docker template)
#
# HF Spaces free tier: 2 vCPU, 16GB RAM, no GPU. Fits our 1.6GB DeBERTa
# model comfortably. Listens on $PORT (default 7860 on HF).
#
# Build:  docker build -t factforge .
# Run:    docker run --rm -p 7860:7860 --env-file .env factforge
# =========================================================================

FROM python:3.12-slim

# System deps: build-essential for some Python wheels, libxml2 for lxml,
# curl for healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libxml2 \
        libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user. HF Spaces uses uid 1000 by convention.
RUN useradd -m -u 1000 user
ENV PATH="/home/user/.local/bin:${PATH}" \
    HF_HOME="/home/user/.cache/huggingface" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # HF Spaces sets PORT; default to 7860 for local docker.
    PORT=7860

# Install uv for dep management
RUN pip install --no-cache-dir uv==0.11.14

WORKDIR /app

# Copy lockfile + pyproject first so dep install is cached separately from src
COPY --chown=user:user pyproject.toml uv.lock* ./

# Install deps (no dev tools in prod image)
RUN uv sync --frozen --no-dev --no-install-project 2>/dev/null || uv sync --no-dev --no-install-project

# Now copy source
COPY --chown=user:user src ./src

# Install the package itself
RUN uv sync --no-dev --frozen 2>/dev/null || uv sync --no-dev

USER user

# Pre-download the NLI model at build time so first-request latency is small.
# Falls back gracefully if HF Hub is unreachable during build (model downloads
# at first inference instead).
RUN uv run python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    name='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'; \
    AutoTokenizer.from_pretrained(name); \
    AutoModelForSequenceClassification.from_pretrained(name); \
    print('NLI model pre-cached')" || echo "[warn] NLI model pre-download skipped (will download at runtime)"

EXPOSE 7860

# HF Spaces sets PORT env var; fall back to 7860 for local docker.
CMD ["sh", "-c", "uv run uvicorn factforge.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
