#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! python3 -c "import uvicorn" >/dev/null 2>&1; then
  echo "uvicorn is not installed. Run: python3 -m pip install -r requirements.txt" >&2
  exit 1
fi

export TIKSEARCHREC_API_CONFIG="${TIKSEARCHREC_API_CONFIG:-configs/app/api_debug.yaml}"

echo "Starting SearchRec-LLM API"
echo "Docs: http://127.0.0.1:8000/docs"
python3 -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --reload
