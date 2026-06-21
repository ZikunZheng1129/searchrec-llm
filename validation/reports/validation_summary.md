# Validation Summary

Current scope: Stage 5 validation over real result CSVs from synthetic/local
debug data. Ranking, LLM, and multimodal rows are included when their result
files exist.

## Input Files

- validation/results/genrec_results.csv
- validation/results/llm_query_understanding_results.csv
- validation/results/multimodal_results.csv
- validation/results/ranking_results.csv
- validation/results/recommender_baselines.csv
- validation/results/retrieval_results.csv
- validation/results/sequence_results.csv

## Generated Outputs

- validation/results/final_leaderboard.csv
- validation/reports/final_model_selection_report.md
- validation/reports/latency_quality_tradeoff.md
- validation/reports/validation_summary.md

## Leaderboard Rows

31

## Best Method Per Current Stage

- genrec: `candidate_constrained_generation` by valid_item_rate=1.0000
- llm_query_understanding: `llm_query_understanding` by schema_valid_rate=1.0000
- multimodal: `text_metadata_fusion` by ndcg_at_10=0.2930
- ranking: `lightgbm` by ndcg_at_10=0.2999
- recommendation: `itemcf` by ndcg_at_10=0.0557
- retrieval: `bm25` by recall_at_50=1.0000
- sequence: `sasrec` by ndcg_at_10=0.0297

## Missing Future Stages

- Ranking rows: included when `validation/results/ranking_results.csv` exists.
- Multimodal rows: included when `validation/results/multimodal_results.csv` exists.
- GenRec rows: included when `validation/results/genrec_results.csv` exists.

## Reproduce

```bash
python3 src/pipelines/generate_validation_report.py \
  --config validation/experiments/stage5_debug_validation.yaml
bash scripts/run_validation.sh
```
