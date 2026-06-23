# Retrieval Tuning Log

This log compares each retrieval tuning step against the fixed BM25 benchmark using the configured synthetic data.

## Dataset Notes

| num_items | num_query_pairs | train_query_pairs | val_query_pairs | test_query_pairs | num_eval_rows | unique_query_texts | repeated_query_text_rate | strict_top_k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 300 | 243 | 31 | 26 | 26 | 21 | 0.1923 | 10 |

Retrieval pass/fail is based on `Recall@50`, because this stage is evaluated as a candidate generator. `Recall@10` is a tie-break signal, and `MRR@10` is kept only as a diagnostic for early ordering.

## Evaluation Change Log

Added multi-relevant retrieval evaluation. The original single-target metrics score each query-item row against exactly one target item. The multi-relevant view groups rows by `query_text`, treats every target item for that text as relevant, and reports candidate-pool coverage at 50. This does not change the retriever; it only clarifies whether broad synthetic queries are being judged too narrowly.

## Step Changes

| step | name | changed_from_previous | item_text_fields | k1 | b | field_weights | boosts | top_k | k_values |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | bm25_default_benchmark | benchmark baseline | title, category, brand, description | 1.5000 | 0.7500 | none | none | 50 | 5, 10, 20, 50 |
| 1 | bm25_mrr_optimized | item_text_fields, b | title, category, brand | 1.5000 | 0.0000 | none | none | 50 | 5, 10, 20, 50 |
| 2 | bm25_field_weighted | item_text_fields, b, field_weights | title, category, brand, description | 1.5000 | 0.2500 | brand=1.0, category=2.0, description=1.0, title=1.0 | none | 50 | 5, 10, 20, 50 |
| 3 | bm25_category_heavy_no_desc | item_text_fields, b, field_weights | title, category, brand | 1.5000 | 0.0000 | brand=1.0, category=3.0, title=1.0 | none | 50 | 5, 10, 20, 50 |
| 4 | bm25_light_description | item_text_fields, field_weights | title, category, brand, description | 1.5000 | 0.0000 | brand=1.0, category=2.0, description=0.25, title=2.0 | none | 50 | 5, 10, 20, 50 |

## Step Results

| step | name | selection | change | threshold | pass_status | recall_at_10 | recall_at_50 | hit_any_at_50 | recall_multi_at_50 | mrr_at_10 | avg_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | bm25_default_benchmark |  | Baseline BM25 over title, category, brand, and description with default length normalization. | single recall_at_50>=0.90 or multi hit_any_at_50>=0.90 and recall_multi_at_50>=0.90; observe recall_at_10>=0.60 | <span style="color: green; font-weight: 600;">pass: single pool</span> | 0.5769 | 1.0000 | 1.0000 | 1.0000 | 0.2288 | 0.0693 |
| 1 | bm25_mrr_optimized |  | Drop description and disable BM25 length normalization to reduce metadata noise. | single recall_at_50>=0.90 or multi hit_any_at_50>=0.90 and recall_multi_at_50>=0.90; observe recall_at_10>=0.60 | <span style="color: green; font-weight: 600;">pass: single pool</span> | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 0.2765 | 0.0599 |
| 2 | bm25_field_weighted | **BEST** | Balance early recall by boosting category while keeping description and length normalization. | single recall_at_50>=0.90 or multi hit_any_at_50>=0.90 and recall_multi_at_50>=0.90; observe recall_at_10>=0.60 | <span style="color: green; font-weight: 600;">pass: single pool</span> | 0.6538 | 1.0000 | 1.0000 | 1.0000 | 0.2374 | 0.0683 |
| 3 | bm25_category_heavy_no_desc |  | Small-data BM25 using title/category/brand only, with category emphasis to recover top-10 recall without description noise. | single recall_at_50>=0.90 or multi hit_any_at_50>=0.90 and recall_multi_at_50>=0.90; observe recall_at_10>=0.60 | <span style="color: green; font-weight: 600;">pass: single pool</span> | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 0.2765 | 0.0612 |
| 4 | bm25_light_description |  | Small-data BM25 keeps description with low weight to test whether light semantic context improves recall without hurting MRR. | single recall_at_50>=0.90 or multi hit_any_at_50>=0.90 and recall_multi_at_50>=0.90; observe recall_at_10>=0.60 | <span style="color: green; font-weight: 600;">pass: single pool</span> | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 0.2765 | 0.0678 |

## Deltas

| step | name | delta_vs_previous_mrr_at_10 | delta_vs_benchmark_mrr_at_10 | delta_vs_benchmark_recall_at_10 | delta_vs_benchmark_recall_at_50 |
| --- | --- | --- | --- | --- | --- |
| 1 | bm25_mrr_optimized | 0.0478 | 0.0478 | -0.0769 | 0.0000 |
| 2 | bm25_field_weighted | -0.0391 | 0.0087 | 0.0769 | 0.0000 |
| 3 | bm25_category_heavy_no_desc | 0.0391 | 0.0478 | -0.0769 | 0.0000 |
| 4 | bm25_light_description | 0.0000 | 0.0478 | -0.0769 | 0.0000 |

## Multi-Relevant Evaluation

| config_name | num_unique_query_texts | avg_relevant_items_per_query_text | max_relevant_items_per_query_text | hit_any_at_50 | recall_multi_at_50 |
| --- | --- | --- | --- | --- | --- |
| bm25_category_heavy_no_desc | 21 | 1.2381 | 2 | 1.0000 | 1.0000 |
| bm25_default_benchmark | 21 | 1.2381 | 2 | 1.0000 | 1.0000 |
| bm25_field_weighted | 21 | 1.2381 | 2 | 1.0000 | 1.0000 |
| bm25_light_description | 21 | 1.2381 | 2 | 1.0000 | 1.0000 |
| bm25_mrr_optimized | 21 | 1.2381 | 2 | 1.0000 | 1.0000 |

`hit_any_at_50` asks whether at least one relevant item for the query text was retrieved in the candidate pool. `recall_multi_at_50` asks what fraction of all known relevant items for that query text were retrieved.

## Evaluation Interpretation

| question | observation | implication |
| --- | --- | --- |
| Does single-target scoring understate retrieval? | benchmark recall_at_50=1.0000; multi hit_any_at_50=1.0000 | Compare these values to see whether single-target labels understate candidate-pool coverage for broad queries. |
| Which config gives the strongest multi-relevant candidate pool? | bm25_field_weighted has recall_multi_at_50=1.0000 and ties for best | Use this as supporting evidence for the selected config, but keep the pass/fail decision tied to the retrieval threshold. |

## Retrieval Decision

| decision_point | evidence | retrieval_takeaway |
| --- | --- | --- |
| selected_config | bm25_field_weighted selected by Recall@50, Recall@10, then latency. | Use this as the retrieval candidate generator for the next stage. |
| candidate_pool | selected Recall@50=1.0000 (bm25_field_weighted) | Top-50 candidate generation passes the single-target pool gate for this 100-item retrieval set. |
| top10_recall | best Recall@10=0.6538 (bm25_field_weighted) | Use Recall@10 as a tie-break among configs that pass Recall@50; it is not the retrieval pass/fail criterion. |
| early_rank | best MRR@10=0.2765 (bm25_mrr_optimized) | Use MRR@10 only as an ordering diagnostic; candidate generation is judged by Recall@50. |
| next_action | All retained BM25 retrieval variants pass the retrieval pool gate. | Choose the candidate generator by pool coverage first, then Recall@10 and latency as tie-breaks. |

## Query Type Breakdown

| config_name | query_type | num_queries | recall_at_10 | mrr_at_10 | avg_target_rank | miss_rate_at_10 |
| --- | --- | --- | --- | --- | --- | --- |
| bm25_category_heavy_no_desc | brand_category | 2 | 1.0000 | 0.7500 | 1.5000 | 0.0000 |
| bm25_category_heavy_no_desc | category | 4 | 0.5000 | 0.2812 | 4.5000 | 0.5000 |
| bm25_category_heavy_no_desc | descriptive | 7 | 0.2857 | 0.1633 | 4.0000 | 0.7143 |
| bm25_category_heavy_no_desc | price_category | 3 | 0.3333 | 0.1667 | 2.0000 | 0.6667 |
| bm25_category_heavy_no_desc | title | 2 | 1.0000 | 0.7500 | 1.5000 | 0.0000 |
| bm25_category_heavy_no_desc | use_case | 8 | 0.5000 | 0.1778 | 6.0000 | 0.5000 |
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
| bm25_light_description | brand_category | 2 | 1.0000 | 0.7500 | 1.5000 | 0.0000 |
| bm25_light_description | category | 4 | 0.5000 | 0.2812 | 4.5000 | 0.5000 |
| bm25_light_description | descriptive | 7 | 0.2857 | 0.1633 | 4.0000 | 0.7143 |
| bm25_light_description | price_category | 3 | 0.3333 | 0.1667 | 2.0000 | 0.6667 |
| bm25_light_description | title | 2 | 1.0000 | 0.7500 | 1.5000 | 0.0000 |
| bm25_light_description | use_case | 8 | 0.5000 | 0.1778 | 6.0000 | 0.5000 |
| bm25_mrr_optimized | brand_category | 2 | 1.0000 | 0.7500 | 1.5000 | 0.0000 |
| bm25_mrr_optimized | category | 4 | 0.5000 | 0.2812 | 4.5000 | 0.5000 |
| bm25_mrr_optimized | descriptive | 7 | 0.2857 | 0.1633 | 4.0000 | 0.7143 |
| bm25_mrr_optimized | price_category | 3 | 0.3333 | 0.1667 | 2.0000 | 0.6667 |
| bm25_mrr_optimized | title | 2 | 1.0000 | 0.7500 | 1.5000 | 0.0000 |
| bm25_mrr_optimized | use_case | 8 | 0.5000 | 0.1778 | 6.0000 | 0.5000 |

## Output Artifacts

- `/Users/jiatonggao/Desktop/searchrec-llm/validation/results/retrieval_tuning_results.csv`
- `/Users/jiatonggao/Desktop/searchrec-llm/validation/results/retrieval_tuning_comparison.csv`
- `/Users/jiatonggao/Desktop/searchrec-llm/validation/results/retrieval_tuning_strict_comparison.csv`
- `/Users/jiatonggao/Desktop/searchrec-llm/validation/results/retrieval_tuning_per_query.csv`
- `/Users/jiatonggao/Desktop/searchrec-llm/validation/results/retrieval_tuning_query_type_breakdown.csv`
- `/Users/jiatonggao/Desktop/searchrec-llm/validation/results/retrieval_tuning_multi_relevant.csv`
- `/Users/jiatonggao/Desktop/searchrec-llm/validation/reports/retrieval_tuning_log.md`
