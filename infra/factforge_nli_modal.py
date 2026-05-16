"""GPU-hosted NLI inference for factforge, served via Modal.

This module defines a Modal application that exposes a single POST
endpoint `/score` for sentence-pair NLI scoring. The factforge backend
(running on HF Spaces free CPU) calls this endpoint to offload the
expensive DeBERTa-v3-large forward passes onto a Modal T4 GPU.

────────────────────────────────────────────────────────────────────
Why this exists
────────────────────────────────────────────────────────────────────
On HF Spaces free CPU, DeBERTa-v3-large NLI (435M params) takes
~3-5 seconds per evidence chunk. With ~24 chunks per claim, the
NLI step alone dominates the response time (~70-120s). On a Modal
T4 GPU, the same model is ~30x faster: ~150ms per chunk, ~3-5s
total for the whole step.

────────────────────────────────────────────────────────────────────
Cost
────────────────────────────────────────────────────────────────────
T4 GPU on Modal: $0.59/hr = $0.000164/sec
- Cold call (model already in container's GPU memory): ~3s ≈ $0.0005
- First cold-start of a container (model load to GPU): adds ~5-10s
- Scaledown after 5 min of idle → container shuts down, no charge
Modal Starter free tier: $1/month, hard-capped (no credit card required)
At our traffic, monthly cost stays ~$0.02-0.10 — effectively free.

────────────────────────────────────────────────────────────────────
Auth
────────────────────────────────────────────────────────────────────
Every request must include the header `X-API-Key: <shared-secret>`.
The secret is injected via Modal Secret `factforge-nli` (env var
`FACTFORGE_NLI_API_KEY`). The factforge backend stores the same
secret as `MODAL_NLI_API_KEY` in its HF Space env.

────────────────────────────────────────────────────────────────────
Deploy
────────────────────────────────────────────────────────────────────
    # One-time, before first deploy:
    KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    modal secret create factforge-nli FACTFORGE_NLI_API_KEY="$KEY"

    # Deploy (rerun any time the code changes):
    modal deploy infra/factforge_nli_modal.py

The deploy step prints a URL of the form:
    https://<workspace>--factforge-nli-web.modal.run
Add that URL + the API key to the factforge backend's env as
`MODAL_NLI_URL` and `MODAL_NLI_API_KEY`.

────────────────────────────────────────────────────────────────────
Failure behaviour
────────────────────────────────────────────────────────────────────
factforge's NLI client is designed remote-first with a local CPU
fallback. If this Modal endpoint is unavailable for any reason
(rate-limited, exhausted budget, network blip, deploy in progress),
factforge transparently falls back to the bundled DeBERTa model
running on HF's CPU. Users never see an error; they just see the
slower path for that window.
"""

from __future__ import annotations

import modal

MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
MAX_LENGTH = 512

app = modal.App("factforge-nli")


# ---------------------------------------------------------------------------
# Image: install deps + pre-download the NLI model at build time so the
# first request doesn't pay a 1.6GB download. The model files end up baked
# into the container image's filesystem (HF cache at ~/.cache/huggingface).
# ---------------------------------------------------------------------------
def _download_model() -> None:
    """Run at image-build time inside the Modal build container."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    AutoTokenizer.from_pretrained(MODEL_NAME)
    AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "transformers>=4.46.0",
        "torch>=2.5.0",
        "accelerate>=1.0.0",
        "sentencepiece>=0.2.0",
        "protobuf>=5.0.0",
        "fastapi[standard]>=0.115.0",
        "pydantic>=2.9.0",
    )
    .run_function(_download_model)
)


# ---------------------------------------------------------------------------
# The actual service.
#
# Using `@modal.asgi_app()` (with a full FastAPI app) instead of
# `@modal.fastapi_endpoint()` gives us native FastAPI Header() injection
# for the API key, plus a free /health endpoint. The function body runs
# ONCE per container start (model load), then FastAPI handles all
# subsequent requests until the container scales down.
# ---------------------------------------------------------------------------
@app.function(
    gpu="T4",
    image=image,
    scaledown_window=300,  # keep the container warm 5 min between bursts
    timeout=120,  # plenty for an NLI batch; protects against runaway calls
    secrets=[modal.Secret.from_name("factforge-nli")],
)
@modal.asgi_app()
def web():
    """Build and return the FastAPI app. Runs once per container start."""
    import os

    import torch
    import torch.nn.functional as F
    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # --- Load model into GPU memory (one-time per container) ---
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # Resolve label indices from model config — order isn't fixed across
    # checkpoints, so don't hardcode integer positions.
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    e_idx = next(i for i, v in id2label.items() if v.startswith("entail"))
    n_idx = next(i for i, v in id2label.items() if v.startswith("neutr"))
    c_idx = next(i for i, v in id2label.items() if v.startswith("contra"))

    expected_api_key = os.environ.get("FACTFORGE_NLI_API_KEY", "")

    # --- Schemas ---
    class ScoreRequest(BaseModel):
        premises: list[str]
        hypothesis: str

    class NLIScore(BaseModel):
        entailment: float
        neutral: float
        contradiction: float

    class ScoreResponse(BaseModel):
        scores: list[NLIScore]

    # --- App ---
    web_app = FastAPI(
        title="factforge-nli",
        description="GPU-hosted DeBERTa-v3-large NLI inference for factforge.",
        version="1.0.0",
    )

    def _check_auth(x_api_key: str) -> None:
        # Reject if the server has no key configured (misconfiguration) OR
        # if the caller sent the wrong one. Same status either way so we
        # don't leak which it is.
        if not expected_api_key or x_api_key != expected_api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

    @web_app.get("/health")
    def health() -> dict[str, str]:
        """Unauthenticated liveness probe (used by warm-up pings)."""
        return {"status": "ok", "model": MODEL_NAME}

    @web_app.post("/score", response_model=ScoreResponse)
    def score(
        req: ScoreRequest,
        x_api_key: str = Header(default="", alias="X-API-Key"),
    ) -> dict[str, list[dict[str, float]]]:
        """Score each premise against the same hypothesis.

        Returns the entailment/neutral/contradiction probabilities for
        each (premise, hypothesis) pair in the same order as `premises`.
        """
        _check_auth(x_api_key)

        if not req.premises or not req.hypothesis:
            return {"scores": []}

        # Batched forward pass on GPU. We tokenize all pairs at once; the
        # transformer handles them in parallel. No need to chunk further
        # for our typical batch sizes (≤ 32).
        with torch.no_grad():
            enc = tokenizer(
                req.premises,
                [req.hypothesis] * len(req.premises),
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
            ).to(device)
            logits = model(**enc).logits
            probs = F.softmax(logits, dim=-1).cpu().tolist()

        return {
            "scores": [
                {
                    "entailment": float(p[e_idx]),
                    "neutral": float(p[n_idx]),
                    "contradiction": float(p[c_idx]),
                }
                for p in probs
            ]
        }

    return web_app
