# API Reference

Local OpenAPI docs are available at `http://127.0.0.1:8000/docs` after running:

```bash
bash scripts/launch_api.sh
```

## Endpoints

- `GET /health`
- `GET /artifacts/status`
- `POST /search`
- `GET /recommend/{user_id}`
- `POST /llm/query-understanding`
- `POST /genrec`
- `GET /models/leaderboard`
- `GET /metrics/business`
- `GET /errors/taxonomy`

## Example Search Request

```json
{
  "query_text": "gift beauty",
  "top_k": 10,
  "use_llm_query_understanding": true,
  "use_multimodal_candidates": true,
  "include_explanations": true
}
```

## Example GenRec Request

```json
{
  "query_text": "gift beauty",
  "top_k": 10,
  "candidate_pool_size": 20,
  "method": "candidate_constrained_generation"
}
```

## Error Format

```json
{
  "error": "artifact_missing",
  "message": "Missing data/processed/ranking_candidates.parquet.",
  "command_hint": "python src/pipelines/build_ranking_dataset.py --config configs/ranking/ranking_dataset_debug.yaml"
}
```

## Notes

The default API uses local artifacts and mock LLM components. No OpenAI key or
external API call is required.
