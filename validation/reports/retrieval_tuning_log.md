# Retrieval Tuning Log

This log compares each retrieval tuning step against the fixed BM25 benchmark using the current synthetic debug data.

## Step Changes

| step | name | changed_from_previous | item_text_fields | k1 | b | field_weights | boosts | top_k | k_values |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | bm25_default_benchmark | benchmark baseline | title, category, brand, description | 1.5000 | 0.7500 | none | none | 50 | 5, 10, 20, 50 |
| 1 | bm25_mrr_optimized | item_text_fields, b | title, category, brand | 1.5000 | 0.0000 | none | none | 50 | 5, 10, 20, 50 |
| 2 | bm25_field_weighted | item_text_fields, b, field_weights | title, category, brand, description | 1.5000 | 0.2500 | brand=1.0, category=2.0, description=1.0, title=1.0 | none | 50 | 5, 10, 20, 50 |
| 3 | query_aware_use_case_boost | method, boosts, use_case_terms | title, category, brand, description | 1.5000 | 0.2500 | brand=1.0, category=2.0, description=1.0, title=1.0 | brand_match_boost=0.0, category_match_boost=0.0, popularity_boost=0.0, rating_boost=0.0, use_case_term_boost=0.35 | 50 | 5, 10, 20, 50 |

## Step Results

| step | name | change | recall_at_5 | recall_at_10 | recall_at_50 | mrr_at_5 | mrr_at_10 | mrr_at_50 | avg_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | bm25_default_benchmark | Baseline BM25 over title, category, brand, and description with default length normalization. | 0.3462 | 0.5769 | 1.0000 | 0.1981 | 0.2288 | 0.2538 | 0.0756 |
| 1 | bm25_mrr_optimized | Drop description and disable BM25 length normalization to reduce metadata noise. | 0.3462 | 0.5000 | 1.0000 | 0.2577 | 0.2765 | 0.3111 | 0.0637 |
| 2 | bm25_field_weighted | Balance early recall by boosting category while keeping description and length normalization. | 0.3462 | 0.6538 | 1.0000 | 0.1981 | 0.2374 | 0.2595 | 0.0755 |
| 3 | query_aware_use_case_boost | Add use-case-triggered text boosts on top of the balanced field-weighted BM25. | 0.4231 | 0.6923 | 1.0000 | 0.2269 | 0.2666 | 0.2851 | 2.1367 |

## Deltas

| step | name | delta_vs_previous_mrr_at_10 | delta_vs_benchmark_mrr_at_10 | delta_vs_benchmark_recall_at_10 | delta_vs_benchmark_recall_at_50 |
| --- | --- | --- | --- | --- | --- |
| 1 | bm25_mrr_optimized | 0.0478 | 0.0478 | -0.0769 | 0.0000 |
| 2 | bm25_field_weighted | -0.0391 | 0.0087 | 0.0769 | 0.0000 |
| 3 | query_aware_use_case_boost | 0.0292 | 0.0378 | 0.1154 | 0.0000 |

## Strict Top-10 Comparison

| candidate_config_path | delta_recall_at_3 | delta_recall_at_5 | delta_recall_at_10 | delta_mrr_at_10 |
| --- | --- | --- | --- | --- |
| /Users/jiatonggao/Desktop/searchrec-llm/configs/retrieval/retrieval_tuning_bm25_mrr_optimized.yaml | 0.0769 | 0.0000 | -0.0769 | 0.0478 |
| /Users/jiatonggao/Desktop/searchrec-llm/configs/retrieval/retrieval_tuning_bm25_field_weighted.yaml | 0.0000 | 0.0000 | 0.0769 | 0.0087 |
| /Users/jiatonggao/Desktop/searchrec-llm/configs/retrieval/retrieval_tuning_query_aware_boost.yaml | 0.0385 | 0.0769 | 0.1154 | 0.0378 |

## Query Type Breakdown

| config_name | query_type | num_queries | recall_at_10 | mrr_at_10 | avg_target_rank | miss_rate_at_10 |
| --- | --- | --- | --- | --- | --- | --- |
| bm25_default_benchmark | brand_category | 2 | 1.0000 | 0.7500 | 1.5000 | 0.0000 |
| bm25_default_benchmark | category | 4 | 0.7500 | 0.1319 | 6.3333 | 0.2500 |
| bm25_default_benchmark | descriptive | 7 | 0.7143 | 0.2315 | 5.6000 | 0.2857 |
| bm25_default_benchmark | price_category | 3 | 0.3333 | 0.1667 | 2.0000 | 0.6667 |
| bm25_default_benchmark | title | 2 | 1.0000 | 0.7500 | 1.5000 | 0.0000 |
| bm25_default_benchmark | use_case | 8 | 0.2500 | 0.0375 | 7.5000 | 0.7500 |
| bm25_field_weighted | brand_category | 2 | 1.0000 | 0.7500 | 1.5000 | 0.0000 |
| bm25_field_weighted | category | 4 | 0.7500 | 0.1319 | 6.3333 | 0.2500 |
| bm25_field_weighted | descriptive | 7 | 0.7143 | 0.2315 | 5.6000 | 0.2857 |
| bm25_field_weighted | price_category | 3 | 0.3333 | 0.1667 | 2.0000 | 0.6667 |
| bm25_field_weighted | title | 2 | 1.0000 | 0.7500 | 1.5000 | 0.0000 |
| bm25_field_weighted | use_case | 8 | 0.5000 | 0.0656 | 8.2500 | 0.5000 |
| bm25_mrr_optimized | brand_category | 2 | 1.0000 | 0.7500 | 1.5000 | 0.0000 |
| bm25_mrr_optimized | category | 4 | 0.5000 | 0.2812 | 4.5000 | 0.5000 |
| bm25_mrr_optimized | descriptive | 7 | 0.2857 | 0.1633 | 4.0000 | 0.7143 |
| bm25_mrr_optimized | price_category | 3 | 0.3333 | 0.1667 | 2.0000 | 0.6667 |
| bm25_mrr_optimized | title | 2 | 1.0000 | 0.7500 | 1.5000 | 0.0000 |
| bm25_mrr_optimized | use_case | 8 | 0.5000 | 0.1778 | 6.0000 | 0.5000 |
| query_aware_use_case_boost | brand_category | 2 | 1.0000 | 0.7500 | 1.5000 | 0.0000 |
| query_aware_use_case_boost | category | 4 | 0.7500 | 0.1319 | 6.3333 | 0.2500 |
| query_aware_use_case_boost | descriptive | 7 | 0.7143 | 0.2315 | 5.6000 | 0.2857 |
| query_aware_use_case_boost | price_category | 3 | 0.3333 | 0.1667 | 2.0000 | 0.6667 |
| query_aware_use_case_boost | title | 2 | 1.0000 | 0.7500 | 1.5000 | 0.0000 |
| query_aware_use_case_boost | use_case | 8 | 0.6250 | 0.1604 | 4.6000 | 0.3750 |

## Error Analysis

### bm25_mrr_optimized Wins At 10

| query_text | query_type | target_item_id | benchmark_rank | candidate_rank | rank_delta |
| --- | --- | --- | --- | --- | --- |
| health and personal care products | category | item_00011 | 0 | 1 | 998 |
| gift health and personal care | use_case | item_00034 | 0 | 5 | 994 |
| school electronics | use_case | item_00037 | 0 | 9 | 990 |

### bm25_mrr_optimized Losses At 10

| query_text | query_type | target_item_id | benchmark_rank | candidate_rank | rank_delta |
| --- | --- | --- | --- | --- | --- |
| popular beauty item | descriptive | item_00075 | 5 | 0 | -994 |
| electronics products | category | item_00054 | 6 | 0 | -993 |
| popular electronics item | descriptive | item_00057 | 7 | 0 | -992 |
| sports and outdoors products | category | item_00058 | 9 | 0 | -990 |
| high rated beauty item | descriptive | item_00064 | 9 | 0 | -990 |

### bm25_field_weighted Wins At 10

| query_text | query_type | target_item_id | benchmark_rank | candidate_rank | rank_delta |
| --- | --- | --- | --- | --- | --- |
| travel electronics | use_case | item_00068 | 0 | 8 | 991 |
| travel electronics | use_case | item_00098 | 0 | 10 | 989 |

### bm25_field_weighted Losses At 10

| query_text | query_type | target_item_id | benchmark_rank | candidate_rank | rank_delta |
| --- | --- | --- | --- | --- | --- |

### query_aware_use_case_boost Wins At 10

| query_text | query_type | target_item_id | benchmark_rank | candidate_rank | rank_delta |
| --- | --- | --- | --- | --- | --- |
| school electronics | use_case | item_00037 | 0 | 4 | 995 |
| walking clothing | use_case | item_00048 | 0 | 6 | 993 |
| travel sports and outdoors | use_case | item_00089 | 0 | 6 | 993 |

### query_aware_use_case_boost Losses At 10

| query_text | query_type | target_item_id | benchmark_rank | candidate_rank | rank_delta |
| --- | --- | --- | --- | --- | --- |

## Output Artifacts

- `validation/results/retrieval_tuning_results.csv`
- `validation/results/retrieval_tuning_comparison.csv`
- `validation/results/retrieval_tuning_strict_comparison.csv`
- `validation/results/retrieval_tuning_per_query.csv`
- `validation/results/retrieval_tuning_query_type_breakdown.csv`
- `validation/reports/retrieval_tuning_log.md`
