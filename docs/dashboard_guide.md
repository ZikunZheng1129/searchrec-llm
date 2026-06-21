# Dashboard Guide

## Pages

- Search Demo: query examples and ranked candidates.
- LLM Query Understanding: mock structured parsing.
- User Profile: synthetic profile summaries and profile-based fallback items.
- Ranking Pipeline: candidate rows and ranking metrics.
- GenRec Demo: candidate-constrained recommendations and explanations.
- Model Comparison: final leaderboard and best method per component.
- Error Analysis: taxonomy and analysis prompts.
- Business Metrics: synthetic proxy metrics.

## Local Artifact Mode

The dashboard uses local parquet, CSV, and markdown artifacts by default. It can
run without the API server.

## API Mode

`configs/app/dashboard_debug.yaml` includes `api_base_url` for future API-backed
views. The current pages default to local artifact mode for reliability.

## Common Missing Files

Use the artifact status table on the landing page. Missing rows include command
hints for the pipeline command that creates the file.

## Technical Demo Flow

1. Open the Search Demo and show ranked candidates.
2. Open LLM Query Understanding and show structured mock output.
3. Open GenRec Demo and point out candidate-constrained valid item IDs.
4. Open Model Comparison and explain component-specific metrics.
5. Open Business Metrics and state that metrics are synthetic proxies.

## Caveat

The dashboard is local-first and uses synthetic debug data. It should not be
described as production-ready.
