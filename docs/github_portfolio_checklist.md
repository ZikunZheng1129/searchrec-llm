# GitHub Portfolio Checklist

## Repository Readiness

- [ ] README is polished and easy to skim.
- [ ] Synthetic debug data caveat is visible.
- [ ] Mock LLM caveat is visible.
- [ ] Final report exists.
- [ ] Results summary uses real validation CSV values.
- [ ] Demo guide, API reference, and dashboard guide exist.
- [ ] Resume bullets and interview materials exist.
- [ ] Tests pass.
- [ ] Ruff passes.
- [ ] Format check passes.
- [ ] No secrets are committed.
- [ ] `.env.example` is safe if present.
- [ ] Generated large artifacts are gitignored or intentionally excluded.
- [ ] Optional screenshots are clearly labeled as local demo screenshots if added later.

## Suggested Artifacts To Keep

- Source code under `src/` and `app/`.
- Configs under `configs/`.
- Validation templates under `validation/experiments/`.
- Small result CSVs under `validation/results/`.
- Reports under `validation/reports/` and `docs/`.
- Tests under `tests/`.

## Suggested Artifacts To Exclude Or Review

- Large checkpoints beyond debug artifacts.
- Large embeddings.
- Raw downloaded datasets.
- Local virtual environments.
- API keys, `.env`, or private notebooks.
- Generated screenshots unless intentionally curated.

## Suggested GitHub Description

Production-style local search and recommendation system with retrieval, ranking, text/metadata representations, LLM query understanding, candidate-constrained GenRec, validation reports, FastAPI, and Streamlit.

## Suggested Topics

- `search`
- `recommendation-system`
- `ranking`
- `llm`
- `genrec`
- `fastapi`
- `streamlit`
- `machine-learning`
- `multimodal`
- `recommender-systems`

## Final Pre-Push Commands

```bash
python3 src/pipelines/generate_validation_report.py --config validation/experiments/stage5_debug_validation.yaml
python3 -m pytest tests/
python3 -m ruff check .
python3 -m ruff format --check .
bash scripts/check_project_package.sh
```
