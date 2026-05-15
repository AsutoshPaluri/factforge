"""Sentence-embedding client for memory-augmented learning.

Uses `sentence-transformers/all-MiniLM-L6-v2` — 22M params, 384-dim
output, fast on CPU (~50ms per claim), 100% local (no API quota).
Auto-downloads ~80MB on first call, cached in ~/.cache/huggingface/.

Used by:
  - `routes.py` to embed each incoming claim before agent run
  - `repositories.py` to store embeddings + find similar past claims

We use the underlying `transformers` library directly (which we already
have for NLI) rather than installing `sentence-transformers` as a
separate dep, to keep the Docker image lean.
"""

from __future__ import annotations

import asyncio
import time
from typing import Sequence

import structlog
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = structlog.get_logger(__name__)


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MAX_LENGTH = 256  # MiniLM max is 512; 256 is plenty for typical claims


class EmbeddingClient:
    """Async wrapper around MiniLM. Mirrors NLI client's lazy-load pattern."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._loaded = False
        self._device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self._load_lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        async with self._load_lock:
            if self._loaded:
                return
            await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        logger.info(
            "embedding_loading_model",
            model=self.model_name,
            device=str(self._device),
        )
        t0 = time.time()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self._device)
        self._model.eval()
        self._loaded = True
        logger.info("embedding_loaded", elapsed_s=round(time.time() - t0, 2))

    def _mean_pool(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean-pooling with attention-mask weighting. Standard for MiniLM."""
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        return summed / counts

    @torch.no_grad()
    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        assert self._tokenizer is not None and self._model is not None
        enc = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        ).to(self._device)

        out = self._model(**enc)
        pooled = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
        # L2-normalize so cosine similarity == dot product downstream.
        normalized = F.normalize(pooled, p=2, dim=1)
        return normalized.cpu().tolist()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return a list of 384-dim L2-normalised vectors, one per input text."""
        await self._ensure_loaded()
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_sync, list(texts))

    async def embed_one(self, text: str) -> list[float]:
        """Convenience: embed a single string."""
        out = await self.embed([text])
        return out[0]

    async def warmup(self) -> None:
        """Load the model now so the first embed call is fast."""
        await self._ensure_loaded()


# --- Process-wide singleton ---
_instance: EmbeddingClient | None = None


def get_embedder() -> EmbeddingClient:
    """Process-wide EmbeddingClient singleton."""
    global _instance
    if _instance is None:
        _instance = EmbeddingClient()
    return _instance
