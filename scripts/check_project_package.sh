#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python3 -m pytest tests/
python3 -m ruff check .
python3 -m ruff format --check .

test -f README.md
test -f docs/final_report.md
test -f docs/final_results_summary.md
test -f docs/project_completion_audit.md
test -f docs/resume_bullets.md
test -f docs/interview_talking_points.md
test -f docs/demo_script.md
test -f validation/results/final_leaderboard.csv

echo "SearchRec-LLM package check passed."
