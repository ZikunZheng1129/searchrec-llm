#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! python3 -c "import uvicorn, streamlit" >/dev/null 2>&1; then
  echo "uvicorn/streamlit missing. Run: python3 -m pip install -r requirements.txt" >&2
  exit 1
fi

echo "Run the demo stack in two terminals:"
echo
echo "Terminal 1:"
echo "  bash scripts/launch_api.sh"
echo
echo "Terminal 2:"
echo "  bash scripts/launch_dashboard.sh"
echo
echo "API docs: http://127.0.0.1:8000/docs"
echo "Dashboard: http://localhost:8501"
