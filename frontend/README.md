# factforge — frontend

Next.js 16 + React 19 + Tailwind 4 UI for the factforge fact-checking agent.

See the [repo-root README](../README.md) for the full project description.

**Live:** [factforge.vercel.app](https://factforge.vercel.app)

## Local development

```bash
cd frontend
npm install

# Point at your local backend (or the live HF Space)
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
#  or, to develop against production backend:
# echo "NEXT_PUBLIC_API_URL=https://AsutoshPaluri-factforge.hf.space" > .env.local

npm run dev  # http://localhost:3000
```

## Stack

- **Framework**: Next.js 16 (App Router, Turbopack)
- **UI**: Tailwind CSS 4 (dark theme, glass-morphism, aurora bg gradient)
- **Language**: TypeScript 5
- **Deploy**: Vercel (Production domain: `factforge.vercel.app`)

## Layout

```
frontend/
├── package.json
├── next.config.ts
├── tailwind.config.ts
├── src/app/
│   ├── layout.tsx          # global shell, fonts, metadata
│   ├── page.tsx            # the single-page UI (claim input, verdict card, feedback)
│   ├── globals.css         # Tailwind v4 base + dark-theme overrides
│   └── api/                # API route handlers (proxy if needed)
└── public/                 # static assets
```

## Notable UI bits

- **Verdict card** — colour-coded by label: emerald (`Credible`) / amber (`Uncertain`) / rose (`Not Credible`)
- **Agent-working timeline** — live progress through the 5 nodes (Decompose, Retrieve, Verify, Synthesize, Summarize)
- **Feedback bar** — 👍/👎 + free-text comment + Refine button (re-runs the agent with the feedback as additional context)
- **Cited summary** — plain-English explanation with inline source links rendered from the agent's structured output

## Deploy

`main` branch auto-deploys to Vercel on push.

The single required env var is `NEXT_PUBLIC_API_URL` (the backend's URL). Set it in Vercel project settings → Environment Variables.
