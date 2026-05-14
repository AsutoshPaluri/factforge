# Deployment guide

Public stack:

- **Backend** → HuggingFace Spaces (Docker, free 16GB RAM, fits the 1.6GB NLI model)
- **Frontend** → Vercel (free Hobby plan)
- **DB** → Supabase (already provisioned)

Result: a free, publicly-accessible URL like `factforge.vercel.app` that talks to `username-factforge.hf.space`.

---

## 1. Deploy the backend to HuggingFace Spaces

### 1a. Create the Space (web UI, ~2 min)

1. Go to https://huggingface.co/new-space
2. Owner: your username
3. Space name: `factforge` (or whatever — this becomes part of the URL)
4. License: MIT
5. SDK: **Docker** (NOT Gradio/Streamlit)
6. Docker template: **Blank**
7. Hardware: **CPU basic** (free)
8. Visibility: Public

Click **Create Space**. You'll land on an empty Space page.

### 1b. Add the API keys as Secrets

In your Space: **Settings → Variables and secrets → New secret** for each of:

| Secret name | Value (from your local `.env`) |
|---|---|
| `GEMINI_API_KEY` | your Gemini key |
| `SUPABASE_URL` | your Supabase URL |
| `SUPABASE_PUBLISHABLE_KEY` | your publishable key |
| `SUPABASE_SECRET_KEY` | your secret key |
| `FRONTEND_URL` | your Vercel URL (set after step 2; can be `*` for now) |

These get injected as env vars at runtime — they're not visible in the repo.

### 1c. Push the backend code

The Space has a git remote. Clone it, copy your backend code in, push:

```bash
# from anywhere outside the factforge repo
git clone https://huggingface.co/spaces/<your-username>/factforge factforge-space
cd factforge-space

# Copy backend code in (Dockerfile must be at the root of the Space)
cp -r /Users/asutoshpaluri/Documents/factforge/backend/* .
cp /Users/asutoshpaluri/Documents/factforge/backend/.dockerignore .

# Add an HF-flavored README.md at the root of the Space repo
cat > README.md <<'EOF'
---
title: factforge
emoji: "🔎"
colorFrom: slate
colorTo: zinc
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# factforge — backend

Multimodal fact-checking agent. See https://github.com/<you>/factforge for the full project.
EOF

git add .
git commit -m "deploy backend"
git push
```

HF will start building the Docker image. First build takes ~5-10 min (installs torch, downloads the NLI model). You'll see logs in the Space's **App** tab. When it says "Running" with a green dot, the endpoint is live:

```
https://<your-username>-factforge.hf.space/api/v1/health
```

Verify with curl. Should return `{"status":"ok","version":"0.1.0"}`.

### 1d. Test the live endpoint

```bash
curl -X POST "https://<your-username>-factforge.hf.space/api/v1/claims" \
  -H "Content-Type: application/json" \
  -d '{"claim": "Humans only use 10 percent of their brain"}'
```

First request might take 30-60s (cold start + model load). Subsequent requests ~10-30s.

---

## 2. Deploy the frontend to Vercel

### 2a. Push the monorepo to GitHub

```bash
cd /Users/asutoshpaluri/Documents/factforge
gh repo create factforge --public --source=. --remote=origin --push
# Or use the GitHub web UI to create the repo and:
#   git remote add origin git@github.com:<you>/factforge.git
#   git push -u origin main
```

### 2b. Import to Vercel

1. Go to https://vercel.com/new
2. Import the `factforge` repo
3. **Root Directory:** `frontend` ← important (monorepo)
4. Framework Preset: Next.js (auto-detected)
5. **Environment Variables:**
   - `NEXT_PUBLIC_API_URL` = `https://<your-username>-factforge.hf.space`
6. Click **Deploy**

Build takes ~2 min. You'll get a URL like `factforge-<hash>.vercel.app`.

### 2c. Lock the backend's CORS to your Vercel URL

Once you know the Vercel URL, update the `FRONTEND_URL` secret in HF Spaces (step 1b) to that URL. Restart the Space (Settings → Restart). Now the backend only accepts requests from your frontend (plus localhost during dev).

---

## 3. Cold-start mitigation

HF Spaces sleeps after 48h of no traffic. First request after sleep ≈ 60-90s
(container start + model load).

**Easy fix:** GitHub Actions cron that pings `/api/v1/health` every 30 min.
Stays inside HF Spaces' free-tier wakeup budget.

```yaml
# .github/workflows/keepalive.yml
name: keepalive
on:
  schedule:
    - cron: "*/30 * * * *"
jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - run: curl -fsS "https://<your-username>-factforge.hf.space/api/v1/health"
```

---

## 4. Troubleshooting

| Symptom | Fix |
|---|---|
| HF build fails on "torch" | Check Space hardware: needs CPU basic (free) or upgrade |
| 401 from Gemini in HF | Re-check `GEMINI_API_KEY` secret value (no quotes, no trailing whitespace) |
| CORS error in browser | `FRONTEND_URL` secret must match the exact Vercel URL (incl. `https://`) |
| Frontend "Failed to fetch" | Open browser devtools network tab. If 404, check `NEXT_PUBLIC_API_URL` in Vercel env vars |
| Verdicts take 60+s | First-request cold start. Add the keepalive cron above. |
