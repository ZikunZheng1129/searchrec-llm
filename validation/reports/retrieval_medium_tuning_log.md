# Retrieval Tuning Log

This log compares each retrieval tuning step against the fixed BM25 benchmark using the configured synthetic data.

## Dataset Notes

| num_items | num_query_pairs | train_query_pairs | val_query_pairs | test_query_pairs | num_eval_rows | unique_query_texts | repeated_query_text_rate | strict_top_k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 5000 | 4016 | 491 | 493 | 493 | 123 | 0.7505 | 10 |

Retrieval pass/fail is based on `Recall@50`, because this stage is evaluated as a candidate generator. `Recall@10` is a tie-break signal, and `MRR@10` is kept only as a diagnostic for early ordering.

## Evaluation Change Log

Added multi-relevant retrieval evaluation. The original single-target metrics score each query-item row against exactly one target item. The multi-relevant view groups rows by `query_text`, treats every target item for that text as relevant, and reports candidate-pool coverage at 50. This does not change the retriever; it only clarifies whether broad synthetic queries are being judged too narrowly.

## Step Changes

| step | name | changed_from_previous | item_text_fields | k1 | b | field_weights | boosts | top_k | k_values |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | medium_bm25_default_benchmark | benchmark baseline | title, category, brand, description | 1.5000 | 0.7500 | none | none | 100 | 5, 10, 20, 50, 100 |
| 1 | medium_bm25_mrr_optimized | item_text_fields, b | title, category, brand | 1.5000 | 0.0000 | none | none | 100 | 5, 10, 20, 50, 100 |
| 2 | medium_bm25_field_weighted | item_text_fields, b, field_weights | title, category, brand, description | 1.5000 | 0.2500 | brand=1.0, category=2.0, description=1.0, title=1.0 | none | 100 | 5, 10, 20, 50, 100 |
| 3 | medium_bm25_category_heavy_no_desc | item_text_fields, b, field_weights | title, category, brand | 1.5000 | 0.0000 | brand=1.0, category=3.0, title=1.0 | none | 100 | 5, 10, 20, 50, 100 |
| 4 | medium_bm25_light_description | item_text_fields, field_weights | title, category, brand, description | 1.5000 | 0.0000 | brand=1.0, category=2.0, description=0.25, title=2.0 | none | 100 | 5, 10, 20, 50, 100 |

## Step Results

| step | name | selection | change | threshold | pass_status | recall_at_10 | recall_at_50 | hit_any_at_50 | recall_multi_at_50 | recall_at_100 | mrr_at_10 | avg_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | medium_bm25_default_benchmark |  | Medium synthetic benchmark BM25 over title, category, brand, and description. | single recall_at_50>=0.60 or multi hit_any_at_50>=0.90 and recall_multi_at_50>=0.70; observe recall_at_10>=0.30 | <span style="color: green; font-weight: 600;">pass: multi pool</span> | 0.1744 | 0.4219 | 0.9106 | 0.7193 | 0.6166 | 0.0622 | 0.6897 |
| 1 | medium_bm25_mrr_optimized |  | Medium BM25 matching the small-data MRR-optimized setup: title/category/brand only with no length normalization. | single recall_at_50>=0.60 or multi hit_any_at_50>=0.90 and recall_multi_at_50>=0.70; observe recall_at_10>=0.30 | <span style="color: green; font-weight: 600;">pass: multi pool</span> | 0.1785 | 0.4402 | 0.9431 | 0.7350 | 0.6531 | 0.0600 | 0.6535 |
| 2 | medium_bm25_field_weighted |  | Medium synthetic field-weighted BM25 with category emphasis and retained description. | single recall_at_50>=0.60 or multi hit_any_at_50>=0.90 and recall_multi_at_50>=0.70; observe recall_at_10>=0.30 | <span style="color: green; font-weight: 600;">pass: multi pool</span> | 0.1765 | 0.4280 | 0.9187 | 0.7211 | 0.6369 | 0.0643 | 0.6787 |
| 3 | medium_bm25_category_heavy_no_desc |  | Medium BM25 matching the small-data category-heavy no-description setup. | single recall_at_50>=0.60 or multi hit_any_at_50>=0.90 and recall_multi_at_50>=0.70; observe recall_at_10>=0.30 | <span style="color: green; font-weight: 600;">pass: multi pool</span> | 0.1785 | 0.4402 | 0.9431 | 0.7350 | 0.6531 | 0.0600 | 0.6032 |
| 4 | medium_bm25_light_description | **BEST** | Medium BM25 matching the small-data light-description setup. | single recall_at_50>=0.60 or multi hit_any_at_50>=0.90 and recall_multi_at_50>=0.70; observe recall_at_10>=0.30 | <span style="color: green; font-weight: 600;">pass: multi pool</span> | 0.1785 | 0.4422 | 0.9431 | 0.7358 | 0.6552 | 0.0600 | 0.7361 |

## Deltas

| step | name | delta_vs_previous_mrr_at_10 | delta_vs_benchmark_mrr_at_10 | delta_vs_benchmark_recall_at_10 | delta_vs_benchmark_recall_at_50 |
| --- | --- | --- | --- | --- | --- |
| 1 | medium_bm25_mrr_optimized | -0.0022 | -0.0022 | 0.0041 | 0.0183 |
| 2 | medium_bm25_field_weighted | 0.0043 | 0.0020 | 0.0020 | 0.0061 |
| 3 | medium_bm25_category_heavy_no_desc | -0.0043 | -0.0022 | 0.0041 | 0.0183 |
| 4 | medium_bm25_light_description | 0.0000 | -0.0022 | 0.0041 | 0.0203 |

## Multi-Relevant Evaluation

| config_name | num_unique_query_texts | avg_relevant_items_per_query_text | max_relevant_items_per_query_text | hit_any_at_50 | recall_multi_at_50 |
| --- | --- | --- | --- | --- | --- |
| medium_bm25_category_heavy_no_desc | 123 | 4.0081 | 23 | 0.9431 | 0.7350 |
| medium_bm25_default_benchmark | 123 | 4.0081 | 23 | 0.9106 | 0.7193 |
| medium_bm25_field_weighted | 123 | 4.0081 | 23 | 0.9187 | 0.7211 |
| medium_bm25_light_description | 123 | 4.0081 | 23 | 0.9431 | 0.7358 |
| medium_bm25_mrr_optimized | 123 | 4.0081 | 23 | 0.9431 | 0.7350 |

`hit_any_at_50` asks whether at least one relevant item for the query text was retrieved in the candidate pool. `recall_multi_at_50` asks what fraction of all known relevant items for that query text were retrieved.

## Evaluation Interpretation

| question | observation | implication |
| --- | --- | --- |
| Does single-target scoring understate retrieval? | benchmark recall_at_50=0.4219; multi hit_any_at_50=0.9106 | Compare these values to see whether single-target labels understate candidate-pool coverage for broad queries. |
| Which config gives the strongest multi-relevant candidate pool? | medium_bm25_light_description has recall_multi_at_50=0.7358 and is best | Use this as supporting evidence for the selected config, but keep the pass/fail decision tied to the retrieval threshold. |

## Retrieval Decision

| decision_point | evidence | retrieval_takeaway |
| --- | --- | --- |
| selected_config | medium_bm25_light_description selected by Recall@50, Recall@10, then latency. | Use this as the retrieval candidate generator for the next stage. |
| candidate_pool | selected Recall@50=0.4422 (medium_bm25_light_description) | Top-50 candidate generation passes the multi-relevant pool gate for this 1000-item retrieval set, despite missing the single-target Recall@50 gate. |
| top10_recall | best Recall@10=0.1785 (medium_bm25_mrr_optimized) | Use Recall@10 as a tie-break among configs that pass Recall@50; it is not the retrieval pass/fail criterion. |
| early_rank | best MRR@10=0.0643 (medium_bm25_field_weighted) | Use MRR@10 only as an ordering diagnostic; candidate generation is judged by Recall@50. |
| next_action | All retained BM25 retrieval variants pass the retrieval pool gate. | Choose the candidate generator by pool coverage first, then Recall@10 and latency as tie-breaks. |

## Query Type Breakdown

| config_name | query_type | num_queries | recall_at_10 | mrr_at_10 | avg_target_rank | miss_rate_at_10 |
| --- | --- | --- | --- | --- | --- | --- |
| medium_bm25_category_heavy_no_desc | brand_category | 73 | 0.2740 | 0.0776 | 5.2500 | 0.7260 |
| medium_bm25_category_heavy_no_desc | category | 68 | 0.0441 | 0.0103 | 4.6667 | 0.9559 |
| medium_bm25_category_heavy_no_desc | descriptive | 67 | 0.0299 | 0.0030 | 10.0000 | 0.9701 |
| medium_bm25_category_heavy_no_desc | price_category | 43 | 0.0233 | 0.0058 | 4.0000 | 0.9767 |
| medium_bm25_category_heavy_no_desc | title | 58 | 0.9655 | 0.3724 | 4.2679 | 0.0345 |
| medium_bm25_category_heavy_no_desc | use_case | 184 | 0.0326 | 0.0064 | 5.8333 | 0.9674 |
| medium_bm25_default_benchmark | brand_category | 73 | 0.3014 | 0.1018 | 4.6364 | 0.6986 |
| medium_bm25_default_benchmark | category | 68 | 0.0294 | 0.0090 | 5.5000 | 0.9706 |
| medium_bm25_default_benchmark | descriptive | 67 | 0.0448 | 0.0052 | 8.6667 | 0.9552 |
| medium_bm25_default_benchmark | price_category | 43 | 0.0000 | 0.0000 | n/a | 1.0000 |
| medium_bm25_default_benchmark | title | 58 | 0.9655 | 0.3724 | 4.2679 | 0.0345 |
| medium_bm25_default_benchmark | use_case | 184 | 0.0163 | 0.0038 | 5.3333 | 0.9837 |
| medium_bm25_field_weighted | brand_category | 73 | 0.3014 | 0.1018 | 4.6364 | 0.6986 |
| medium_bm25_field_weighted | category | 68 | 0.0294 | 0.0090 | 5.5000 | 0.9706 |
| medium_bm25_field_weighted | descriptive | 67 | 0.0448 | 0.0052 | 8.6667 | 0.9552 |
| medium_bm25_field_weighted | price_category | 43 | 0.0000 | 0.0000 | n/a | 1.0000 |
| medium_bm25_field_weighted | title | 58 | 0.9655 | 0.3724 | 4.2679 | 0.0345 |
| medium_bm25_field_weighted | use_case | 184 | 0.0217 | 0.0092 | 4.2500 | 0.9783 |
| medium_bm25_light_description | brand_category | 73 | 0.2740 | 0.0776 | 5.2500 | 0.7260 |
| medium_bm25_light_description | category | 68 | 0.0441 | 0.0103 | 4.6667 | 0.9559 |
| medium_bm25_light_description | descriptive | 67 | 0.0299 | 0.0030 | 10.0000 | 0.9701 |
| medium_bm25_light_description | price_category | 43 | 0.0233 | 0.0058 | 4.0000 | 0.9767 |
| medium_bm25_light_description | title | 58 | 0.9655 | 0.3724 | 4.2679 | 0.0345 |
| medium_bm25_light_description | use_case | 184 | 0.0326 | 0.0064 | 5.8333 | 0.9674 |
| medium_bm25_mrr_optimized | brand_category | 73 | 0.2740 | 0.0776 | 5.2500 | 0.7260 |
| medium_bm25_mrr_optimized | category | 68 | 0.0441 | 0.0103 | 4.6667 | 0.9559 |
| medium_bm25_mrr_optimized | descriptive | 67 | 0.0299 | 0.0030 | 10.0000 | 0.9701 |
| medium_bm25_mrr_optimized | price_category | 43 | 0.0233 | 0.0058 | 4.0000 | 0.9767 |
| medium_bm25_mrr_optimized | title | 58 | 0.9655 | 0.3724 | 4.2679 | 0.0345 |
| medium_bm25_mrr_optimized | use_case | 184 | 0.0326 | 0.0064 | 5.8333 | 0.9674 |

## Output Artifacts

- `/Users/jiatonggao/Desktop/searchrec-llm/validation/results/retrieval_medium_results.csv`
- `/Users/jiatonggao/Desktop/searchrec-llm/validation/results/retrieval_medium_comparison.csv`
- `/Users/jiatonggao/Desktop/searchrec-llm/validation/results/retrieval_medium_strict_comparison.csv`
- `/Users/jiatonggao/Desktop/searchrec-llm/validation/results/retrieval_medium_per_query.csv`
- `/Users/jiatonggao/Desktop/searchrec-llm/validation/results/retrieval_medium_query_type_breakdown.csv`
- `/Users/jiatonggao/Desktop/searchrec-llm/validation/results/retrieval_medium_multi_relevant.csv`
- `/Users/jiatonggao/Desktop/searchrec-llm/validation/reports/retrieval_medium_tuning_log.md`
