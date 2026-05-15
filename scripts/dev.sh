#!/usr/bin/env bash
# factforge — start both backend and frontend dev servers.
#
# Usage:
#   ./scripts/dev.sh
#
# Runs backend in the background (logs to logs/backend.log) and
# frontend in the foreground. Ctrl+C stops both cleanly.

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

# Load tool managers (uv, nvm) since cron / non-interactive shells lack them
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" >/dev/null 2>&1

if [ ! -f "$ROOT/.env" ]; then
  echo "[dev.sh] No .env at $ROOT/.env — copy .env.example and fill it in first."
  exit 1
fi

# --- Backend (background) ---
echo "[dev.sh] Starting backend on http://localhost:8000 (logs: $LOG_DIR/backend.log)"
(
  cd "$ROOT/backend"
  uv run uvicorn factforge.main:app \
    --host 127.0.0.1 --port 8000 --reload \
    >"$LOG_DIR/backend.log" 2>&1
) &
BACKEND_PID=$!

# Clean up backend on exit (Ctrl+C, error, etc.)
cleanup() {
  echo ""
  echo "[dev.sh] Stopping backend (pid $BACKEND_PID)..."
  kill "$BACKEND_PID" 2>/dev/null || true
  wait "$BACKEND_PID" 2>/dev/null || true
  echo "[dev.sh] Done."
}
trap cleanup EXIT INT TERM

# Wait for backend health endpoint to be ready before starting frontend
echo "[dev.sh] Waiting for backend health (~10s — loads 1.6GB NLI model)..."
SECONDS_WAITED=0
until curl -sf http://127.0.0.1:8000/api/v1/health >/dev/null 2>&1; do
  sleep 1
  SECONDS_WAITED=$((SECONDS_WAITED + 1))
  if [ $SECONDS_WAITED -gt 60 ]; then
    echo "[dev.sh] Backend didn't come up in 60s. Check $LOG_DIR/backend.log:"
    tail -20 "$LOG_DIR/backend.log"
    exit 1
  fi
done
echo "[dev.sh] Backend ready (${SECONDS_WAITED}s)"

# --- Frontend (foreground) ---
echo "[dev.sh] Starting frontend on http://localhost:3000"
echo "[dev.sh] Open http://localhost:3000 in your browser. Ctrl+C to stop both."
echo ""
cd "$ROOT/frontend"
npm run dev
