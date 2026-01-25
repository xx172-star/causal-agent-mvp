#!/usr/bin/env bash
set -euo pipefail

# Run from repo root
cd "$(dirname "$0")/.."

# Require OpenAI key if you want LLM routing
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[WARN] OPENAI_API_KEY is not set. LLM router will fallback to rule-based."
else
  echo "[OK] OPENAI_API_KEY is set."
fi

echo "[START] uvicorn on http://127.0.0.1:8000"
exec uvicorn src.agent.app:app --port 8000
