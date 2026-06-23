# Latency And Quality Tradeoff

Data caveat: all current measurements use the local synthetic debug dataset
and a small item catalog.

## Retrieval

| stage | method | primary_metric_name | primary_score | avg_latency_ms | p95_latency_ms | coverage_score |
| --- | --- | --- | --- | --- | --- | --- |
| retrieval | bm25 | recall_at_50 | 1.0000 | 0.0974 | 0.1360 | 1.0000 |
| retrieval | faiss | recall_at_50 | 1.0000 | 0.0623 | 0.0670 | 1.0000 |
| retrieval | dense | recall_at_50 | 1.0000 | 0.0695 | 0.0709 | 1.0000 |
| retrieval | hybrid | recall_at_50 | 1.0000 | 0.2786 | 0.3077 | 1.0000 |

## Recommendation

| stage | method | primary_metric_name | primary_score | avg_latency_ms | p95_latency_ms | coverage_score |
| --- | --- | --- | --- | --- | --- | --- |
| recommendation | itemcf | ndcg_at_10 | 0.0557 | 0.0568 | 0.0693 | 0.9800 |
| recommendation | matrix_factorization | ndcg_at_10 | 0.0404 | 0.0438 | 0.0516 | 0.7900 |
| recommendation | user_history_embedding | ndcg_at_10 | 0.0369 | 0.0524 | 0.0556 | 0.9900 |
| recommendation | popularity | ndcg_at_10 | 0.0321 | 0.0332 | 0.0377 | 0.1500 |

## Sequence

| stage | method | primary_metric_name | primary_score | avg_latency_ms | p95_latency_ms | coverage_score |
| --- | --- | --- | --- | --- | --- | --- |
| sequence | sasrec | ndcg_at_10 | 0.0297 | 0.3631 | 0.3798 | 0.6700 |
| sequence | gru4rec | ndcg_at_10 | 0.0230 | 0.1826 | 0.2005 | 0.6100 |

## Ranking

| stage | method | primary_metric_name | primary_score | avg_latency_ms | p95_latency_ms | coverage_score |
| --- | --- | --- | --- | --- | --- | --- |
| ranking | lightgbm | ndcg_at_10 | 0.2999 | 0.1215 | 0.1215 | 1.0000 |
| ranking | mixed | ndcg_at_10 | 0.2718 | 2.4340 | 2.4340 | 1.0000 |
| ranking | mlp | ndcg_at_10 | 0.2713 | 0.1657 | 0.1657 | 1.0000 |
| ranking | cross_encoder | ndcg_at_10 | 0.1327 | 4.8991 | 4.8991 | 1.0000 |

## LLM Query Understanding

| stage | method | primary_metric_name | primary_score | avg_latency_ms | p95_latency_ms | coverage_score |
| --- | --- | --- | --- | --- | --- | --- |
| llm_query_understanding | llm_query_understanding | schema_valid_rate | 1.0000 | 0.0983 | 0.1480 | 1.0000 |

## Multimodal

| stage | method | primary_metric_name | primary_score | avg_latency_ms | p95_latency_ms | coverage_score |
| --- | --- | --- | --- | --- | --- | --- |
| multimodal | text_metadata_fusion | ndcg_at_10 | 0.2930 | 0.0655 | 0.0699 | 0.6500 |
| multimodal | text_only | ndcg_at_10 | 0.2801 | 0.0600 | 0.0649 | 0.6800 |
| multimodal | multimodal_fusion | ndcg_at_10 | 0.2454 | 0.0652 | 0.0675 | 0.6500 |
| multimodal | metadata_only | ndcg_at_10 | 0.0530 | 0.0334 | 0.0377 | 0.1000 |

## GenRec

| stage | method | primary_metric_name | primary_score | avg_latency_ms | p95_latency_ms | coverage_score |
| --- | --- | --- | --- | --- | --- | --- |
| genrec | candidate_constrained_generation | valid_item_rate | 1.0000 | 0.0591 | 0.0787 | 1.0000 |
| genrec | candidate_constrained_generation | valid_item_rate | 1.0000 | 0.0635 | 0.0642 | 1.0000 |
| genrec | llm_rerank_top_10 | valid_item_rate | 1.0000 | 0.0387 | 0.0439 | 1.0000 |
| genrec | llm_rerank_top_10 | valid_item_rate | 1.0000 | 0.0475 | 0.0481 | 1.0000 |
| genrec | llm_rerank_top_20 | valid_item_rate | 1.0000 | 0.0566 | 0.0610 | 1.0000 |
| genrec | llm_rerank_top_20 | valid_item_rate | 1.0000 | 0.0569 | 0.0610 | 1.0000 |
| genrec | direct_generation | valid_item_rate | 0.9000 | 0.0194 | 0.0205 | 1.0000 |
| genrec | direct_generation | valid_item_rate | 0.9000 | 0.0198 | 0.0203 | 1.0000 |
| genrec | template_explanation | valid_item_rate | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| genrec | template_explanation | valid_item_rate | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| genrec | evidence_grounded_llm_explanation | valid_item_rate | 0.0000 | 0.0434 | 0.0533 | 1.0000 |
| genrec | evidence_grounded_llm_explanation | valid_item_rate | 0.0000 | 0.0453 | 0.0500 | 1.0000 |

## How To Interpret

Higher quality metrics are better, but latency and coverage make the choice
more practical. A baseline with slightly lower quality may still be valuable
when it is simpler, faster, or covers more of the catalog.

## Current Limitations

- Local synthetic dataset.
- Small item catalog.
- No production load testing yet.

## Future Work

- Larger candidate pools.
- Real FAISS indexes.
- Neural embeddings.
- Ranker latency.
- API latency.
