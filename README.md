# factforge

> Multimodal fact-checking agent. Submit a text claim with an optional image, get an evidence-grounded verdict with full reasoning trace.

![Status](https://img.shields.io/badge/status-in_development-yellow)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Next.js](https://img.shields.io/badge/Next.js-15-black)

**Live demo:** *coming soon*

---

## What it does

Given a claim — *"Vaccines cause autism"* or a screenshot of a viral tweet — factforge runs a multi-stage agentic pipeline:

1. **Decompose** the claim into atomic, independently verifiable sub-claims
2. **Retrieve** evidence from the open web (DuckDuckGo, no API key)
3. **Verify** each sub-claim against retrieved evidence using a fine-tuned NLI model
4. **Weight** evidence by source credibility (domain reputation, recency, primary vs. secondary source)
5. **Synthesize** a final verdict — **Real / Misinformation / Disinformation** — with citations

Multimodal: image inputs (charts, screenshots, photos) are parsed by a vision model to extract the underlying claim, then verified the same way.

## Architecture

```
                ┌─────────────────────────────────┐
                │   Text claim + optional image   │
                └────────────────┬────────────────┘
                                 │
        ┌────────────────────────▼────────────────────────┐
        │ [1] Claim Decomposer  (Gemini 2.5 Flash)        │
        │     → atomic sub-claims                         │
        └────────────────────────┬────────────────────────┘
                                 │
        ┌────────────────────────▼────────────────────────┐
        │ [2] Evidence Retrieval  (DuckDuckGo, no key)    │
        │     → top-k snippets per sub-claim              │
        └────────────────────────┬────────────────────────┘
                                 │
        ┌────────────────────────▼────────────────────────┐
        │ [3] NLI Verifier  (DeBERTa-v3-large-MNLI-FEVER) │
        │     → entailment / contradiction / neutral      │
        └────────────────────────┬────────────────────────┘
                                 │
        ┌────────────────────────▼────────────────────────┐
        │ [4] Source Credibility Scorer                   │
        │     → weighted evidence aggregation             │
        └────────────────────────┬────────────────────────┘
                                 │
        ┌────────────────────────▼────────────────────────┐
        │ [5] Verdict Synthesizer                         │
        │     → 3-class verdict + reasoning + citations   │
        └─────────────────────────────────────────────────┘
```

## Stack

| Layer | Choice |
|---|---|
| Backend | FastAPI · LangGraph · Python 3.12 (uv) |
| Reasoning LLM | Gemini 2.5 Flash (free tier) |
| NLI | DeBERTa-v3-large-MNLI-FEVER (embedded, CPU) |
| Web search | DuckDuckGo (no API key) |
| Database | Supabase (Postgres + pgvector + Auth) |
| Frontend | Next.js 15 · Tailwind · shadcn/ui |
| Observability | Langfuse |
| Deploy | Vercel · HuggingFace Spaces / Fly.io |

**Operating cost: $0/month.** Engineered around free-tier ceilings with hard kill switches.

## Eval

Benchmarked against:

| Benchmark | Type | Status |
|---|---|---|
| FEVER | text claims | TBD |
| AVeriTeC | real-world text claims w/ evidence | TBD |
| MOCHEG | multimodal claims | TBD |

## Development

```bash
git clone https://github.com/<you>/factforge.git
cd factforge

# Backend
cd backend
uv sync
cp ../.env.example .env  # fill in your free-tier keys
uv run uvicorn factforge.main:app --reload

# Frontend (in another shell)
cd frontend
npm install
npm run dev
```

## License

MIT — see [LICENSE](LICENSE).

---

Built by [Asutosh Paluri](https://github.com/asutoshpaluri) · MS Computational Linguistics, UNT (2026)
