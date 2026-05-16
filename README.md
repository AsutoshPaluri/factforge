# factforge

> Multimodal fact-checking agent. Submit a text claim (with an optional image), get an evidence-grounded verdict with full reasoning trace and source citations.

[![Live](https://img.shields.io/badge/live-factforge.vercel.app-brightgreen)](https://factforge.vercel.app)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black)](https://nextjs.org/)

**Live:** [factforge.vercel.app](https://factforge.vercel.app)  ·  **Author portfolio:** [asutoshpaluri.netlify.app](https://asutoshpaluri.netlify.app)  ·  **Cost:** $0/month, no credit card on file

---

## What it does

Given a claim — *"Humans only use 10% of their brain"* or a screenshot of a viral tweet — factforge runs a 5-node agentic pipeline:

1. **Decompose** the claim into atomic, independently verifiable sub-claims (LLM)
2. **Retrieve** open-web evidence for each sub-claim (search + page extraction)
3. **Verify** each (evidence, sub-claim) pair using a fine-tuned NLI model (GPU)
4. **Synthesize** a hybrid binary-with-abstain verdict from per-pair entailment scores
5. **Summarize** the reasoning in plain English with inline source citations

The user can mark the result correct or wrong; that feedback is embedded and stored, then automatically retrieved on similar future claims to improve the decomposition step. It's a Level-2 learning agent (memory-augmented improvement loop), not just a stateless RAG pipeline.

**Verdict labels:** `Credible` · `Uncertain` (abstain) · `Not Credible`

Multimodal: when an image is uploaded, a vision model extracts the latent claim from the picture (chart, screenshot, photo), then the same verification pipeline runs.

## Performance

| Metric | Value |
|---|---|
| End-to-end latency (warm) | **~7-10 seconds per claim** |
| End-to-end latency (cold container) | ~25-30 seconds |
| FEVER binary accuracy (Credible vs Not Credible, N=30 balanced) | **90.0%** |
| FEVER 3-class accuracy (incl. NEI → Uncertain) | 73.3% |
| Operating cost | **$0/month** |

The single biggest latency win came from offloading NLI from HuggingFace's free CPU (~70-120s per claim) to a Modal-hosted T4 GPU (~1-3s per claim) — a ~30× speedup on the dominant pipeline step. The local CPU model is retained as a graceful fallback: if Modal is rate-limited, unavailable, or the monthly budget is exhausted, the system transparently falls back without user-visible errors.

## Architecture

factforge is a distributed system across **7 free-tier services**, each handling one job:

```
                              USER
                                │
                                ▼
                  ┌────────────────────────────┐
                  │  VERCEL — Frontend         │
                  │  Next.js 16 + Tailwind 4   │
                  │  factforge.vercel.app      │
                  └──────────────┬─────────────┘
                                 │ POST /api/v1/claims
                                 ▼
              ┌──────────────────────────────────────┐
              │  HUGGINGFACE SPACES — Backend        │
              │  FastAPI + LangGraph orchestrator    │
              │  AsutoshPaluri-factforge.hf.space    │
              │                                      │
              │   [1] Decompose ──▶ [2] Retrieve ──▶ │
              │   [3] Verify  ──▶ [4] Synthesize ──▶ │
              │   [5] Summarize                      │
              │                                      │
              │  Local DeBERTa CPU model (fallback)  │
              └─┬──────┬───────┬──────┬──────┬───────┘
                │      │       │      │      │
                ▼      ▼       ▼      ▼      ▼
            ┌──────┐┌──────┐┌──────┐┌──────┐┌──────────┐
            │MODAL ││GROQ  ││GEMINI││DDGS  ││ SUPABASE │
            │T4 GPU││Llama ││Vision││Search││ Postgres │
            │ NLI  ││3.1-8b││ 2.5  ││ (no  ││ + pgvec  │
            │      ││instnt││Flash ││ key) ││ + Auth   │
            └──────┘└──────┘└──────┘└──────┘└──────────┘

                LANGFUSE — agent traces
                SENTRY   — error reporting
                GITHUB   — source
```

### The agent graph (LangGraph)

```
                ┌─────────────────────────────────┐
                │   Text claim + optional image   │
                └────────────────┬────────────────┘
                                 │
        ┌────────────────────────▼────────────────────────┐
        │ [1] Decomposer  (Groq Llama 3.1 / Gemini-vision)│
        │     → 1-3 atomic sub-claims                     │
        │     + injects relevant past feedback if any     │
        └────────────────────────┬────────────────────────┘
                                 │
        ┌────────────────────────▼────────────────────────┐
        │ [2] Retriever  (DuckDuckGo + trafilatura)       │
        │     → k=8 snippets per sub-claim                │
        │     + domain blocklist, evidence file cache     │
        └────────────────────────┬────────────────────────┘
                                 │
        ┌────────────────────────▼────────────────────────┐
        │ [3] NLI Verifier  (DeBERTa-v3-large MNLI)       │
        │     ├ Modal T4 GPU (primary, ~1-3s)             │
        │     └ HF CPU local model (fallback, ~70-120s)   │
        │     → entailment / neutral / contradiction      │
        └────────────────────────┬────────────────────────┘
                                 │
        ┌────────────────────────▼────────────────────────┐
        │ [4] Synthesizer  (hybrid binary+abstain logic)  │
        │     → Credible / Uncertain / Not Credible       │
        │       + confidence score                        │
        └────────────────────────┬────────────────────────┘
                                 │
        ┌────────────────────────▼────────────────────────┐
        │ [5] Summarizer  (Groq Llama 3.1)                │
        │     → plain-English verdict + cited sources     │
        └────────────────────────┬────────────────────────┘
                                 │
                  ┌──────────────▼──────────────┐
                  │  User: 👍 / 👎 + comment    │
                  │  Stored as pgvector embed,  │
                  │  retrieved on similar future│
                  │  claims (Level-2 learning)  │
                  └─────────────────────────────┘
```

## Stack

| Layer | Choice | Free-tier ceiling |
|---|---|---|
| Frontend | Next.js 16 · React 19 · Tailwind 4 | Vercel Hobby |
| Backend | FastAPI · LangGraph · Python 3.12 (uv) | HuggingFace Spaces CPU |
| NLI verification | DeBERTa-v3-large-MNLI-FEVER-ANLI | Modal T4 GPU ($30/mo compute) |
| Reasoning LLM | Groq Llama 3.1 8b Instant | 500k tokens/day |
| Vision LLM | Gemini 2.5 Flash | 1M tokens/day |
| Web search | DuckDuckGo (`ddgs`) | No API key |
| Page extraction | trafilatura + httpx | Self-hosted |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (384d) | Self-hosted |
| Database | Supabase (Postgres + pgvector + Auth) | 500MB |
| Observability | Langfuse (LangChain CallbackHandler) | 50k traces/month |
| Error tracking | Sentry | 5k events/month |

**$0/month total operating cost. No credit card required on any tier** (Modal's $30/month free compute requires a CC for verification only — set a spend cap at $30 so the card is never charged).

## Engineering decisions worth noting

- **Hybrid binary-with-abstain verdict** — most fact-checking literature uses 3 labels (Real/Misinfo/Disinfo or Supports/Refutes/NEI). Real users want a definitive answer when possible. The synthesizer commits to `Credible` or `Not Credible` unless either (a) the neutral probability dominates entailment+contradiction, or (b) the top binary probability is <0.40 with margin <0.10 — then it abstains as `Uncertain`. Gets a clean 90% binary accuracy on FEVER while honestly admitting ignorance when evidence is genuinely missing.
- **Remote-first NLI with circuit-breaker fallback** — Modal is the primary path; on any failure, a 60-second cooldown is set and traffic routes to the local HF CPU model. The local model lazy-loads on first fallback, so when Modal works, no memory is spent on a redundant model.
- **Memory-augmented learning loop** — feedback isn't just logged; it's embedded with MiniLM and stored in pgvector. On every new claim, the top-K most similar past-feedback entries are retrieved and prepended to the decomposer prompt as soft guidance. This is the Russell-Norvig definition of a Level-2 learning agent.
- **Source quality filtering** — the retriever maintains a domain blocklist (low-credibility outlets) and filters out junk before NLI ever runs, so the GPU isn't wasted scoring entailment against tabloid clickbait.

## Eval

| Benchmark | Accuracy | Notes |
|---|---|---|
| FEVER (binary, N=30 balanced) | **90.0%** | Credible vs Not Credible only; NEI excluded |
| FEVER (3-class, N=30 balanced) | 73.3% | NEI → Uncertain mapping; abstain rate 23% |

Run the FEVER eval yourself: `python eval/run_fever.py --n 30 --balanced`.

## Local development

```bash
git clone https://github.com/AsutoshPaluri/factforge.git
cd factforge
cp .env.example .env  # fill in your free-tier keys (all guides in the file)

# Backend
cd backend
uv sync
uv run uvicorn factforge.main:app --reload  # http://localhost:8000/docs

# Frontend (in another shell)
cd frontend
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
npm run dev  # http://localhost:3000
```

## Deploy

- **Backend → HuggingFace Spaces:** `./scripts/deploy-hf.sh`
- **Frontend → Vercel:** push to `main`; Vercel auto-deploys
- **GPU NLI → Modal (optional):** `modal deploy infra/factforge_nli_modal.py`

Full deploy walkthrough including the env-var checklist for each surface is in `scripts/deploy-hf.sh` and the corresponding Vercel project settings.

## License

MIT — see [LICENSE](LICENSE).

---

Built by [Asutosh Paluri](https://asutoshpaluri.netlify.app) — MS Computational Linguistics, University of North Texas (May 2026) · [GitHub](https://github.com/AsutoshPaluri) · [Portfolio](https://asutoshpaluri.netlify.app)
