#!/usr/bin/env bash
# =========================================================================
# Deploy factforge backend to HuggingFace Spaces.
#
# One-time setup (do this first, in the browser):
#   1. Go to https://huggingface.co/new-space
#   2. Owner: AsutoshPaluri,  Space name: factforge
#   3. License: MIT,  SDK: Docker (blank template)
#   4. Hardware: CPU basic (free),  Visibility: Public
#   5. Click "Create Space"
#
# One-time auth (do this once, then never again):
#   - Generate an HF access token at https://huggingface.co/settings/tokens
#     (Type: "Write")
#   - First time you run this script, git push will prompt:
#       Username: AsutoshPaluri
#       Password: <paste the token>
#     macOS keychain will save it for future runs.
#
# Run:
#   ./scripts/deploy-hf.sh
# =========================================================================

set -e

HF_USERNAME="${HF_USERNAME:-AsutoshPaluri}"
SPACE_NAME="${SPACE_NAME:-factforge}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPACE_DIR="$ROOT/../factforge-space"

echo "[deploy-hf] Target: huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
echo "[deploy-hf] Working in: $SPACE_DIR"
echo ""

# --- Step 1: clone or pull the Space repo ---
if [ ! -d "$SPACE_DIR" ]; then
  echo "[deploy-hf] Cloning Space repo (first run)..."
  git clone "https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}" "$SPACE_DIR"
else
  echo "[deploy-hf] Pulling latest Space state..."
  (cd "$SPACE_DIR" && git pull --rebase || true)
fi

# --- Step 2: rsync backend contents into the Space ---
echo "[deploy-hf] Syncing backend code into Space..."
rsync -av --delete \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  --exclude='.ruff_cache' \
  --exclude='.mypy_cache' \
  --exclude='.pyright_cache' \
  --exclude='tests' \
  --exclude='*.log' \
  --exclude='scripts/smoke_keys.py' \
  --exclude='README.md' \
  "$ROOT/backend/" "$SPACE_DIR/"

# Copy .dockerignore (rsync skips dotfiles starting with . unless explicit)
cp "$ROOT/backend/.dockerignore" "$SPACE_DIR/.dockerignore" 2>/dev/null || true

# --- Step 3: write HF-flavored README at Space root ---
# HF requires colorTo to be one of: red, yellow, green, blue, indigo, purple,
# pink, gray. And short_description must be <= 60 characters.
cat > "$SPACE_DIR/README.md" << 'EOF'
---
title: factforge
emoji: "🔎"
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Multimodal fact-checking AI agent with citations
---

# factforge — backend

Multimodal fact-checking agent. Evidence-grounded NLI verification with
human-in-the-loop refinement and memory-augmented learning.

**Stack:** FastAPI · LangGraph · Groq (Llama 3.1) · Gemini Vision · DuckDuckGo · DeBERTa-MNLI-FEVER · Supabase pgvector

**Endpoints:**
- `POST /api/v1/claims` — submit a claim, get a verdict
- `POST /api/v1/claims/{id}/feedback` — record user feedback
- `POST /api/v1/claims/{id}/refine` — re-run with feedback as context
- `GET /api/v1/health` — health probe
- `GET /docs` — OpenAPI / Swagger UI

**Verdict labels:** `Credible` · `Uncertain` (abstain) · `Not Credible`

**FEVER eval (N=30 balanced):** 90% binary accuracy on Credible vs Not Credible.

Full project: https://github.com/asutoshpaluri/factforge
EOF

# --- Step 4: commit (if changes) + push (always — covers retries) ---
cd "$SPACE_DIR"
git add .

if git diff --cached --quiet; then
  echo "[deploy-hf] No new file changes — checking for unpushed commits..."
else
  git commit -m "Deploy backend $(date +%Y-%m-%d-%H%M)"
fi

# Always push. If everything is already on remote, this is a no-op.
# If a previous run committed locally but the push was rejected
# (e.g. bad YAML), this retries the push with the corrected files.
echo "[deploy-hf] Pushing to HuggingFace..."
git push

echo ""
echo "[deploy-hf] ✓ Deployed."
echo ""
echo "Watch build logs:"
echo "  https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
echo ""
echo "First build takes ~5-10 min (installs torch + transformers, downloads"
echo "DeBERTa-NLI 1.6GB + MiniLM 80MB)."
echo ""
echo "Backend will be live at:"
echo "  https://${HF_USERNAME}-${SPACE_NAME}.hf.space"
echo ""
echo "Don't forget to set secrets in the Space's Settings tab:"
echo "  - GEMINI_API_KEY"
echo "  - GROQ_API_KEY"
echo "  - SUPABASE_URL"
echo "  - SUPABASE_PUBLISHABLE_KEY"
echo "  - SUPABASE_SECRET_KEY"
echo "  - LANGFUSE_HOST"
echo "  - LANGFUSE_PUBLIC_KEY"
echo "  - LANGFUSE_SECRET_KEY"
echo "  - MODAL_NLI_URL          (e.g. https://<user>--factforge-nli-web.modal.run)"
echo "  - MODAL_NLI_API_KEY      (from ~/.factforge_modal_key)"
echo ""
echo "Copy each value from your local .env at:"
echo "  /Users/asutoshpaluri/Documents/factforge/.env"
