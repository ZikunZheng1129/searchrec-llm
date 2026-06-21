# Final Results Summary

This summary uses existing validation CSVs from the local synthetic debug dataset. It is not production performance evidence and does not use platform-internal data. LLM-related rows use deterministic mock clients by default.

## Current Best Rows By Component

| Component | Selected method | Selection reason |
| --- | --- | --- |
| Retrieval | `bm25` | Recall@50 ties at 1.000000, and BM25 has the best MRR@10 at 0.228785. |
| Recommendation | `itemcf` | Highest NDCG@10 at 0.055705. |
| Sequence | `sasrec` | Highest NDCG@10 at 0.029743. |
| Ranking | `lightgbm` with `numpy_linear` backend | Highest NDCG@10 at 0.299915. |
| LLM query understanding | `llm_query_understanding` mock | Schema valid rate 1.000000 and intent match rate 1.000000 on local synthetic labels. |
| Text/metadata representation | `text_metadata_fusion` | Highest NDCG@10 at 0.292996. |
| GenRec | `candidate_constrained_generation` | Valid item rate 1.000000, hallucination rate 0.000000, and NDCG@10 0.245409. |

## Retrieval

| method | recall_at_50 | mrr_at_10 | query_coverage | avg_latency_ms | index_backend |
| --- | --- | --- | --- | --- | --- |
| bm25 | 1.000000 | 0.228785 | 1.000000 | 0.097376 | local |
| dense | 1.000000 | 0.211966 | 1.000000 | 0.069524 | numpy |
| faiss | 1.000000 | 0.211966 | 1.000000 | 0.062269 | numpy |
| hybrid | 1.000000 | 0.211966 | 1.000000 | 0.278596 | bm25+dense_numpy |

BM25 is selected because all methods reached Recall@50 1.000000, but BM25 had the strongest MRR@10.

## Recommendation

| method | ndcg_at_10 | recall_at_10 | mrr_at_10 | coverage_at_10 | avg_latency_ms |
| --- | --- | --- | --- | --- | --- |
| itemcf | 0.055705 | 0.120000 | 0.036937 | 0.980000 | 0.056784 |
| matrix_factorization | 0.040356 | 0.080000 | 0.029000 | 0.790000 | 0.043752 |
| popularity | 0.032091 | 0.080000 | 0.017833 | 0.150000 | 0.033216 |
| user_history_embedding | 0.036852 | 0.100000 | 0.018381 | 0.990000 | 0.052444 |

itemCF is selected for the current synthetic debug recommendation task because it has the highest NDCG@10 and strong coverage.

## Sequence

| method | ndcg_at_10 | recall_at_10 | mrr_at_10 | coverage_at_10 | avg_latency_ms |
| --- | --- | --- | --- | --- | --- |
| gru4rec | 0.022976 | 0.060000 | 0.012024 | 0.610000 | 0.182551 |
| sasrec | 0.029743 | 0.060000 | 0.020000 | 0.670000 | 0.363133 |

SASRec is selected because it has higher NDCG@10 and MRR@10 than GRU4Rec on the debug split.

## Ranking

| method | backend | ndcg_at_10 | mrr_at_10 | recall_at_10 | auc | avg_latency_ms |
| --- | --- | --- | --- | --- | --- | --- |
| cross_encoder | tiny_torch_transformer | 0.132739 | 0.070299 | 0.346154 | 0.642661 | 4.899082 |
| lightgbm | numpy_linear | 0.299915 | 0.227442 | 0.538462 | 0.819889 | 0.159090 |
| mixed | deterministic_mixed | 0.271769 | 0.188507 | 0.538462 | 0.801941 | 2.433997 |
| mlp | torch_mlp | 0.271298 | 0.191819 | 0.538462 | 0.838365 | 0.165707 |

The LightGBM-style ranker with the `numpy_linear` fallback backend is selected because it has the highest NDCG@10 and MRR@10.

## LLM Query Understanding And User Profiles

| component | provider | primary metric | value | note |
| --- | --- | --- | --- | --- |
| query understanding | mock | schema_valid_rate | 1.000000 | intent_match_rate is 1.000000; category_match_rate is 0.923077; brand_match_rate is 0.153846. |
| user profile generation | mock | profile_generation_success_rate | 1.000000 | schema_valid_rate and embedding_coverage are both 1.000000. |

These metrics validate structured local behavior with mock clients. They do not measure real LLM quality.

## Text/Metadata Representation

| method | ndcg_at_10 | recall_at_10 | cold_start_recall_at_10 | catalog_coverage_at_10 | avg_latency_ms |
| --- | --- | --- | --- | --- | --- |
| text_metadata_fusion | 0.292996 | 0.538462 | 0.444444 | 0.650000 | 0.065521 |
| text_only | 0.280077 | 0.500000 | 0.333333 | 0.680000 | 0.059969 |
| multimodal_fusion | 0.245409 | 0.500000 | 0.444444 | 0.650000 | 0.065245 |
| metadata_only | 0.052964 | 0.115385 | 0.222222 | 0.100000 | 0.033394 |

`text_metadata_fusion` is selected because it has the highest NDCG@10 on the current data. The current debug dataset has no real image features, so the current result is local text/metadata fusion plus an optional image interface.

## GenRec Direct Versus Candidate-Constrained

| method | valid_item_rate | hallucination_rate | output_parse_success_rate | ndcg_at_10 | mrr_at_10 | recall_at_10 | avg_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_constrained_generation | 1.000000 | 0.000000 | 1.000000 | 0.245409 | 0.167399 | 0.500000 | 0.059067 |
| direct_generation | 0.900000 | 0.100000 | 1.000000 | 0.045688 | 0.024786 | 0.115385 | 0.019381 |
| llm_rerank_top_10 | 1.000000 | 0.000000 | 1.000000 | 0.241029 | 0.162012 | 0.500000 | 0.038683 |
| llm_rerank_top_20 | 1.000000 | 0.000000 | 1.000000 | 0.239028 | 0.152778 | 0.538462 | 0.056631 |

Candidate-constrained GenRec is the preferred design because it grounds the generated answer in known catalog candidates and avoids the invalid-item risk observed in direct generation.

## API And Dashboard Status

Local demo smoke checks verified:

- FastAPI `/health` returned a healthy response.
- Streamlit launched locally and returned HTTP 200.
- API and dashboard read local artifacts by default.
- No external API key is required for the default mock/local demo.

## Source Files

- `validation/results/final_leaderboard.csv`
- `validation/results/retrieval_results.csv`
- `validation/results/recommender_baselines.csv`
- `validation/results/sequence_results.csv`
- `validation/results/ranking_results.csv`
- `validation/results/llm_query_understanding_results.csv`
- `validation/results/user_profile_results.csv`
- `validation/results/multimodal_results.csv`
- `validation/results/genrec_results.csv`
