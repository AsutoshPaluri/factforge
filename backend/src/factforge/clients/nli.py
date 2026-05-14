"""NLI inference wrapper — DeBERTa-v3-large-MNLI-FEVER (async-compatible).

The model is loaded lazily on first call and reused across requests via
a process-wide singleton. Inference runs inside `asyncio.to_thread` so
the FastAPI event loop stays responsive.

The model's label order isn't fixed across checkpoints, so we resolve
the entailment / neutral / contradiction indices from `config.id2label`
at load time rather than hardcoding integer positions.

Usage:
    verifier = await get_verifier()
    scores = await verifier.score(
        premises=["Earth is approximately spherical.",
                  "Earth is flat."],
        hypothesis="The earth is round.",
    )
    # -> [NLIScore(entailment=0.87, ...),
    #     NLIScore(contradiction=0.92, ...)]
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Sequence

import structlog
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from factforge.config import settings

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class NLIScore:
    """Three-way NLI probabilities for one (premise, hypothesis) pair."""

    entailment: float
    neutral: float
    contradiction: float

    @property
    def non_neutral(self) -> float:
        """Entailment + contradiction — useful as an 'informativeness' signal."""
        return self.entailment + self.contradiction


def _resolve_device(device_setting: str) -> torch.device:
    """Resolve device with sensible fallbacks.

    - 'cuda' -> cuda if available, else mps, else cpu
    - 'mps'  -> mps if available, else cpu
    - 'cpu'  -> cpu
    """
    s = device_setting.lower()
    if s == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if s in {"cuda", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NLIVerifier:
    """NLI inferencer with batched inference and lazy model loading.

    Designed for use inside an async server: load once per process, then
    each `.score()` call runs the forward pass in a thread so the
    asyncio loop isn't blocked.
    """

    DEFAULT_BATCH_SIZE = 8

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name or settings.nli_model_name
        self.device = _resolve_device(device or settings.nli_device)
        self.max_length = max_length
        self.batch_size = batch_size

        # Loaded lazily on first score() call
        self._tokenizer = None
        self._model = None
        self._e_idx: int | None = None
        self._n_idx: int | None = None
        self._c_idx: int | None = None
        self._loaded = False
        self._load_lock = asyncio.Lock()

    # --- Loading ---
    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        async with self._load_lock:
            if self._loaded:
                return
            await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        logger.info(
            "nli_loading_model",
            model=self.model_name,
            device=str(self.device),
        )
        t0 = time.time()

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self.device)
        self._model.eval()

        # Resolve label indices from the model config
        id2label = {
            int(k): v.lower() for k, v in self._model.config.id2label.items()
        }
        self._e_idx = next(i for i, v in id2label.items() if v.startswith("entail"))
        self._n_idx = next(i for i, v in id2label.items() if v.startswith("neutr"))
        self._c_idx = next(i for i, v in id2label.items() if v.startswith("contra"))

        self._loaded = True
        logger.info(
            "nli_loaded",
            elapsed_s=round(time.time() - t0, 2),
            params_m=round(sum(p.numel() for p in self._model.parameters()) / 1e6, 1),
        )

    # --- Inference ---
    @torch.no_grad()
    def _score_batch_sync(
        self, premises: list[str], hypothesis: str
    ) -> list[NLIScore]:
        """Run batched forward passes; one hypothesis paired with each premise."""
        # Loader runs first, so these are guaranteed populated:
        tok = self._tokenizer
        mdl = self._model
        assert tok is not None and mdl is not None
        assert (
            self._e_idx is not None
            and self._n_idx is not None
            and self._c_idx is not None
        )

        results: list[NLIScore] = []
        for start in range(0, len(premises), self.batch_size):
            batch = premises[start : start + self.batch_size]
            enc = tok(
                batch,
                [hypothesis] * len(batch),
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            ).to(self.device)

            logits = mdl(**enc).logits
            probs = F.softmax(logits, dim=-1).cpu().tolist()

            for p in probs:
                results.append(
                    NLIScore(
                        entailment=float(p[self._e_idx]),
                        neutral=float(p[self._n_idx]),
                        contradiction=float(p[self._c_idx]),
                    )
                )
        return results

    async def score(
        self, premises: Sequence[str], hypothesis: str
    ) -> list[NLIScore]:
        """Score each premise against the same hypothesis.

        Returns NLIScore list in the same order as `premises`. Runs in a
        thread to keep the event loop free.
        """
        await self._ensure_loaded()
        if not premises:
            return []
        return await asyncio.to_thread(
            self._score_batch_sync, list(premises), hypothesis
        )

    async def score_one(self, premise: str, hypothesis: str) -> NLIScore:
        """Convenience: single-pair scoring."""
        results = await self.score([premise], hypothesis)
        return results[0]

    async def warmup(self) -> None:
        """Load the model now (so the first scoring call doesn't pay it).

        Safe to call multiple times — subsequent calls are no-ops.
        """
        await self._ensure_loaded()


# --- Process-wide singleton ---
_instance: NLIVerifier | None = None
_instance_lock = asyncio.Lock()


async def get_verifier() -> NLIVerifier:
    """Get the shared NLIVerifier (creates on first call)."""
    global _instance
    if _instance is not None:
        return _instance
    async with _instance_lock:
        if _instance is None:
            _instance = NLIVerifier()
        return _instance
