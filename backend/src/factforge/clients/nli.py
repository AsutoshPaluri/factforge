"""NLI inference wrapper — DeBERTa-v3-large-MNLI-FEVER.

Two backends, transparently composed:

  * **Remote (preferred)** — a Modal-hosted GPU instance of the same
    model. Configured via the `MODAL_NLI_URL` and `MODAL_NLI_API_KEY`
    env vars. Roughly 30× faster per batch than the CPU path
    (~150ms/chunk vs ~3-5s/chunk).

  * **Local (fallback)** — the embedded transformers model running on
    whatever device pytorch picks (CPU on HF free tier). Always
    available. Used when the remote backend is unavailable, hasn't
    been configured, or is in cooldown after a recent failure.

Design notes:
  - The local model loads lazily. If remote works, the local model
    stays unloaded — ~2GB of RAM is never claimed.
  - A 60-second circuit-breaker cooldown is applied after a remote
    failure, so subsequent requests in that window skip the remote
    call entirely (avoiding the per-request timeout penalty).
  - The model's label order isn't fixed across checkpoints, so we
    resolve entailment / neutral / contradiction indices from
    `config.id2label` at local-load time rather than hardcoding.

Usage:
    verifier = await get_verifier()
    scores = await verifier.score(
        premises=["Earth is approximately spherical.",
                  "Earth is flat."],
        hypothesis="The earth is round.",
    )
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Sequence

import httpx
import structlog
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from factforge.config import settings

logger = structlog.get_logger(__name__)


# Modal cold-start can hit ~10-15s; give it room before falling back.
# Still way under the ~70-120s a fully-local NLI step would cost.
_REMOTE_TIMEOUT_S = 30.0

# After a remote failure, skip remote for this long. Keeps slow-failure
# requests rare instead of every-other-request.
_REMOTE_COOLDOWN_S = 60.0


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
    """NLI scorer with optional Modal-GPU primary path.

    On each `.score()` call:
      1. If a remote (Modal) backend is configured and not in cooldown,
         try it first. On success, return its results.
      2. On any failure (timeout, non-2xx, malformed response), log a
         warning, mark remote as in cooldown, and fall through to local.
      3. Local backend lazy-loads the model on first need.
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

        # --- Local model (loaded lazily) ---
        self._tokenizer = None
        self._model = None
        self._e_idx: int | None = None
        self._n_idx: int | None = None
        self._c_idx: int | None = None
        self._loaded = False
        self._load_lock = asyncio.Lock()

        # --- Remote (Modal) ---
        self._remote_url: str = (
            settings.modal_nli_url.rstrip("/") if settings.modal_nli_url else ""
        )
        self._remote_api_key: str = settings.modal_nli_api_key
        self._remote_enabled = bool(self._remote_url and self._remote_api_key)
        self._remote_cooldown_until: float = 0.0
        self._http_client: httpx.AsyncClient | None = None

        if self._remote_enabled:
            logger.info("nli_remote_configured", url=self._remote_url)
        else:
            logger.info(
                "nli_remote_not_configured",
                reason="MODAL_NLI_URL or MODAL_NLI_API_KEY missing — using local model",
            )

    # ------------------------------------------------------------------
    # Local backend
    # ------------------------------------------------------------------
    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        async with self._load_lock:
            if self._loaded:
                return
            await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        logger.info(
            "nli_loading_local_model",
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
            "nli_local_loaded",
            elapsed_s=round(time.time() - t0, 2),
            params_m=round(sum(p.numel() for p in self._model.parameters()) / 1e6, 1),
        )

    @torch.no_grad()
    def _score_batch_local_sync(
        self, premises: list[str], hypothesis: str
    ) -> list[NLIScore]:
        """Run batched forward passes on the local model."""
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

    async def _score_local(
        self, premises: list[str], hypothesis: str
    ) -> list[NLIScore]:
        await self._ensure_loaded()
        return await asyncio.to_thread(
            self._score_batch_local_sync, premises, hypothesis
        )

    # ------------------------------------------------------------------
    # Remote backend (Modal)
    # ------------------------------------------------------------------
    def _remote_available(self) -> bool:
        return self._remote_enabled and time.time() >= self._remote_cooldown_until

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=_REMOTE_TIMEOUT_S)
        return self._http_client

    async def _score_remote(
        self, premises: list[str], hypothesis: str
    ) -> list[NLIScore]:
        """Call the Modal-hosted NLI endpoint. Raises on any failure."""
        client = await self._get_http_client()
        response = await client.post(
            f"{self._remote_url}/score",
            headers={"X-API-Key": self._remote_api_key},
            json={"premises": premises, "hypothesis": hypothesis},
        )
        response.raise_for_status()
        data = response.json()

        scores_raw = data.get("scores")
        if not isinstance(scores_raw, list) or len(scores_raw) != len(premises):
            raise ValueError(
                f"unexpected response shape: got {type(scores_raw).__name__} "
                f"with {len(scores_raw) if isinstance(scores_raw, list) else '?'} "
                f"items, expected list of {len(premises)}"
            )
        return [
            NLIScore(
                entailment=float(s["entailment"]),
                neutral=float(s["neutral"]),
                contradiction=float(s["contradiction"]),
            )
            for s in scores_raw
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def score(
        self, premises: Sequence[str], hypothesis: str
    ) -> list[NLIScore]:
        """Score each premise against the same hypothesis.

        Tries the remote (Modal) backend first if configured and not in
        cooldown. Falls through to the local CPU model on any failure.
        Returns NLIScore list in the same order as `premises`.
        """
        if not premises:
            return []
        prem_list = list(premises)

        if self._remote_available():
            t0 = time.time()
            try:
                scores = await self._score_remote(prem_list, hypothesis)
                logger.info(
                    "nli_remote_ok",
                    n=len(prem_list),
                    elapsed_s=round(time.time() - t0, 2),
                )
                return scores
            except Exception as e:
                self._remote_cooldown_until = (
                    time.time() + _REMOTE_COOLDOWN_S
                )
                logger.warning(
                    "nli_remote_failed_falling_back_to_local",
                    error=str(e),
                    cooldown_s=_REMOTE_COOLDOWN_S,
                )

        t0 = time.time()
        scores = await self._score_local(prem_list, hypothesis)
        logger.info(
            "nli_local_ok",
            n=len(prem_list),
            elapsed_s=round(time.time() - t0, 2),
        )
        return scores

    async def score_one(self, premise: str, hypothesis: str) -> NLIScore:
        """Convenience: single-pair scoring."""
        results = await self.score([premise], hypothesis)
        return results[0]

    async def warmup(self) -> None:
        """Pre-warm whichever backend is configured.

        If remote is configured, hit its /health to wake the container.
        Modal cold-starts include image pull + model load to GPU (~15-25s
        on a fresh container), so we give the ping 45s before timing out.
        Even if the timeout fires, the HTTP request continues server-side
        and warms the container for the next real call — we just lose the
        ability to log a clean 'warmup_ok' for this boot.

        If remote isn't configured (or warmup ping fails outright), we
        eagerly load the local model so the first /claims call doesn't
        pay the ~7s load time. Safe to call multiple times.
        """
        if self._remote_enabled:
            try:
                client = await self._get_http_client()
                response = await client.get(
                    f"{self._remote_url}/health", timeout=45.0
                )
                response.raise_for_status()
                logger.info("nli_remote_warmup_ok")
                return
            except Exception as e:
                # repr() catches httpx exceptions whose str() is empty
                # (e.g. ReadTimeout) so we get a useful log line.
                logger.warning("nli_remote_warmup_failed", error=repr(e))
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
