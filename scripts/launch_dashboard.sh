#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! python3 -c "import streamlit" >/dev/null 2>&1; then
  echo "streamlit is not installed. Run: python3 -m pip install -r requirements.txt" >&2
  exit 1
fi

export TIKSEARCHREC_DASHBOARD_CONFIG="${TIKSEARCHREC_DASHBOARD_CONFIG:-configs/app/dashboard_debug.yaml}"

echo "Starting SearchRec-LLM dashboard"
echo "Dashboard: http://localhost:8501"
python3 -m streamlit run app/dashboard/streamlit_app.py
