# Final Model Selection Report

Stage 5 status: validation framework added for current local baselines.

Data caveat: all current metrics come from synthetic/local debug data.
These results are useful for model-selection mechanics and portfolio discussion,
not production claims.

## Existing Evaluated Components

- Retrieval baselines from Stage 3.
- Recommendation baselines from Stage 4.
- Sequence baselines from Stage 6, when `sequence_results.csv` exists.
- Ranking baselines from Stage 7, when `ranking_results.csv` exists.
- LLM query-understanding outputs from Stage 8, when results exist.
- Multimodal item representations from Stage 9, when `multimodal_results.csv` exists.
- GenRec, LLM reranking, and explanations from Stage 10, when `genrec_results.csv` exists.

## Final Leaderboard

| stage | method | primary_metric_name | primary_score | secondary_metric_name | secondary_score | coverage_metric_name | coverage_score | avg_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| genrec | candidate_constrained_generation | valid_item_rate | 1.0000 | ndcg_at_10 | 0.2454 | output_parse_success_rate | 1.0000 | 0.0591 |
| genrec | candidate_constrained_generation | valid_item_rate | 1.0000 | ndcg_at_10 | 0.2454 | output_parse_success_rate | 1.0000 | 0.0635 |
| genrec | llm_rerank_top_10 | valid_item_rate | 1.0000 | ndcg_at_10 | 0.2410 | output_parse_success_rate | 1.0000 | 0.0387 |
| genrec | llm_rerank_top_10 | valid_item_rate | 1.0000 | ndcg_at_10 | 0.2410 | output_parse_success_rate | 1.0000 | 0.0475 |
| genrec | llm_rerank_top_20 | valid_item_rate | 1.0000 | ndcg_at_10 | 0.2390 | output_parse_success_rate | 1.0000 | 0.0566 |
| genrec | llm_rerank_top_20 | valid_item_rate | 1.0000 | ndcg_at_10 | 0.2390 | output_parse_success_rate | 1.0000 | 0.0569 |
| genrec | direct_generation | valid_item_rate | 0.9000 | ndcg_at_10 | 0.0457 | output_parse_success_rate | 1.0000 | 0.0194 |
| genrec | direct_generation | valid_item_rate | 0.9000 | ndcg_at_10 | 0.0457 | output_parse_success_rate | 1.0000 | 0.0198 |
| genrec | template_explanation | valid_item_rate | 0.0000 | ndcg_at_10 | 0.0000 | output_parse_success_rate | 1.0000 | 0.0000 |
| genrec | template_explanation | valid_item_rate | 0.0000 | ndcg_at_10 | 0.0000 | output_parse_success_rate | 1.0000 | 0.0000 |
| genrec | evidence_grounded_llm_explanation | valid_item_rate | 0.0000 | ndcg_at_10 | 0.0000 | output_parse_success_rate | 1.0000 | 0.0434 |
| genrec | evidence_grounded_llm_explanation | valid_item_rate | 0.0000 | ndcg_at_10 | 0.0000 | output_parse_success_rate | 1.0000 | 0.0453 |
| llm_query_understanding | llm_query_understanding | schema_valid_rate | 1.0000 | intent_match_rate | 1.0000 | output_parse_success_rate | 1.0000 | 0.0983 |
| multimodal | text_metadata_fusion | ndcg_at_10 | 0.2930 | cold_start_recall_at_10 | 0.4444 | catalog_coverage_at_10 | 0.6500 | 0.0655 |
| multimodal | text_only | ndcg_at_10 | 0.2801 | cold_start_recall_at_10 | 0.3333 | catalog_coverage_at_10 | 0.6800 | 0.0600 |
| multimodal | multimodal_fusion | ndcg_at_10 | 0.2454 | cold_start_recall_at_10 | 0.4444 | catalog_coverage_at_10 | 0.6500 | 0.0652 |
| multimodal | metadata_only | ndcg_at_10 | 0.0530 | cold_start_recall_at_10 | 0.2222 | catalog_coverage_at_10 | 0.1000 | 0.0334 |
| ranking | lightgbm | ndcg_at_10 | 0.2999 | mrr_at_10 | 0.2274 | candidate_coverage | 1.0000 | 0.1591 |
| ranking | mixed | ndcg_at_10 | 0.2718 | mrr_at_10 | 0.1885 | candidate_coverage | 1.0000 | 2.4340 |
| ranking | mlp | ndcg_at_10 | 0.2713 | mrr_at_10 | 0.1918 | candidate_coverage | 1.0000 | 0.1657 |
| ranking | cross_encoder | ndcg_at_10 | 0.1327 | mrr_at_10 | 0.0703 | candidate_coverage | 1.0000 | 4.8991 |
| recommendation | itemcf | ndcg_at_10 | 0.0557 | recall_at_10 | 0.1200 | coverage_at_10 | 0.9800 | 0.0568 |
| recommendation | matrix_factorization | ndcg_at_10 | 0.0404 | recall_at_10 | 0.0800 | coverage_at_10 | 0.7900 | 0.0438 |
| recommendation | user_history_embedding | ndcg_at_10 | 0.0369 | recall_at_10 | 0.1000 | coverage_at_10 | 0.9900 | 0.0524 |
| recommendation | popularity | ndcg_at_10 | 0.0321 | recall_at_10 | 0.0800 | coverage_at_10 | 0.1500 | 0.0332 |
| retrieval | bm25 | recall_at_50 | 1.0000 | mrr_at_10 | 0.2288 | query_coverage | 1.0000 | 0.0974 |
| retrieval | faiss | recall_at_50 | 1.0000 | mrr_at_10 | 0.2120 | query_coverage | 1.0000 | 0.0623 |
| retrieval | dense | recall_at_50 | 1.0000 | mrr_at_10 | 0.2120 | query_coverage | 1.0000 | 0.0695 |
| retrieval | hybrid | recall_at_50 | 1.0000 | mrr_at_10 | 0.2120 | query_coverage | 1.0000 | 0.2786 |
| sequence | sasrec | ndcg_at_10 | 0.0297 | recall_at_10 | 0.0600 | coverage_at_10 | 0.6700 | 0.3631 |
| sequence | gru4rec | ndcg_at_10 | 0.0230 | recall_at_10 | 0.0600 | coverage_at_10 | 0.6100 | 0.1826 |

## Best Retrieval Baseline

`bm25` selected for retrieval because it has the best recall_at_50=1.0000, with mrr_at_10=0.2288, query_coverage=1.0000, and avg latency 0.0974 ms.

## Best Recommendation Baseline

`itemcf` selected for recommendation because it has the best ndcg_at_10=0.0557, with recall_at_10=0.1200, coverage_at_10=0.9800, and avg latency 0.0568 ms.

## Best Sequence Baseline

`sasrec` selected for sequence because it has the best ndcg_at_10=0.0297, with recall_at_10=0.0600, coverage_at_10=0.6700, and avg latency 0.3631 ms.

## Best Ranking Baseline

`lightgbm` selected for ranking because it has the best ndcg_at_10=0.2999, with mrr_at_10=0.2274, candidate_coverage=1.0000, and avg latency 0.1591 ms.

## Best LLM Query-Understanding Baseline

`llm_query_understanding` selected for llm_query_understanding because it has the best schema_valid_rate=1.0000, with intent_match_rate=1.0000, output_parse_success_rate=1.0000, and avg latency 0.0983 ms.

## Best Multimodal Baseline

`text_metadata_fusion` selected for multimodal because it has the best ndcg_at_10=0.2930, with cold_start_recall_at_10=0.4444, catalog_coverage_at_10=0.6500, and avg latency 0.0655 ms.

## Best GenRec Baseline

`candidate_constrained_generation` selected for genrec because it has the best valid_item_rate=1.0000, with ndcg_at_10=0.2454, output_parse_success_rate=1.0000, and avg latency 0.0591 ms.

## Tradeoffs

- Quality: selected separately per stage because retrieval and recommendation
  metrics are not directly comparable.
- Coverage: tracked so high quality does not hide narrow catalog exposure.
- Latency: tracked to keep local baselines honest before heavier models arrive.
- Simplicity: simple baselines are preferred when quality is tied or close.

## Not Evaluated Yet

- Real Amazon Reviews data.

## Future Validation Plan

Add future stage result CSVs into `validation/results/`, normalize them into
the same leaderboard schema, and keep model selection stage-specific.

## Interview Talking Point

“We separated unit tests from model validation. Tests prove the code works;
validation compares model choices and tradeoffs.”
