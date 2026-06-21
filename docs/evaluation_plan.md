# Evaluation Plan

SearchRec-LLM treats validation as a first-class project layer. Unit tests prove that code paths work; validation compares model and system choices with standardized metrics, latency, coverage, and caveats.

## Tests Versus Validation

Tests answer: "Does this implementation behave correctly and deterministically?"

Validation answers: "Which baseline is strongest for this component, what does it cost in latency, and what tradeoffs should guide the next model choice?"

Both are required. A model can pass tests and still be the wrong model choice. A validation result can look promising and still be unreliable if the data is synthetic or the evaluation slice is too small.

## Current Validation Result Files

- `validation/results/retrieval_results.csv`
- `validation/results/recommender_baselines.csv`
- `validation/results/sequence_results.csv`
- `validation/results/ranking_results.csv`
- `validation/results/llm_query_understanding_results.csv`
- `validation/results/user_profile_results.csv`
- `validation/results/multimodal_results.csv`
- `validation/results/genrec_results.csv`
- `validation/results/final_leaderboard.csv`

## Metrics By Component

| Component | Metrics |
| --- | --- |
| Retrieval | Recall@K, MRR@K, query coverage, average latency, p95 latency. |
| Recommendation | HitRate@K, Recall@K, NDCG@K, MRR@K, catalog coverage, latency. |
| Sequence | HitRate@K, Recall@K, NDCG@K, MRR@K, coverage, latency. |
| Ranking | NDCG@K, MRR@K, Precision@K, Recall@K, AUC, candidate coverage, latency. |
| LLM query understanding | Parse success, schema validity, intent/category/brand/price/use-case match rates, cost, latency. |
| User profiles | Generation success, schema validity, non-empty profile rate, embedding coverage, profile length, cost, latency. |
| Text/metadata representation | Recall@K, NDCG@K, MRR@K, cold-start recall, long-tail coverage, catalog coverage, category diversity, latency. |
| GenRec | Valid item rate, hallucination rate, parse success, schema validity, fallback rate, NDCG@K, MRR@K, Recall@K, explanation faithfulness, cost, latency. |
| API/dashboard | Health checks, endpoint behavior, artifact availability, smoke-test status. |

## Selection Rules

Each component uses a component-specific primary metric:

- Retrieval: prioritize `recall_at_50`, then `mrr_at_10`, then coverage, then lower latency.
- Recommendation and sequence: prioritize `ndcg_at_10`, then `recall_at_10`, then coverage, then lower latency.
- Ranking: prioritize `ndcg_at_10`, then `mrr_at_10`, then candidate coverage, then lower latency.
- LLM query understanding: prioritize `schema_valid_rate`, then `intent_match_rate`, then parse success, then lower latency/cost.
- Text/metadata representation: prioritize `ndcg_at_10`, then cold-start recall, then catalog coverage, then lower latency.
- GenRec: prioritize `valid_item_rate`, then `ndcg_at_10`, then parse success, then lower hallucination/fallback/latency.

The selected row is only "best" within its own component. Retrieval scores, recommendation scores, ranking scores, and GenRec scores are not globally comparable because they answer different evaluation questions.

## Final Leaderboard Generation

`src/pipelines/generate_validation_report.py` loads `validation/experiments/stage5_debug_validation.yaml` and calls the report utilities in `src/evaluation/report.py`. The report layer reads existing result CSVs, normalizes them into a shared schema, and writes `validation/results/final_leaderboard.csv`.

No fake metrics are created. If a future results file is absent, it is skipped instead of fabricated.

## Synthetic-Data Caveat

All current metrics are computed on a small synthetic debug dataset with deterministic mock LLM behavior by default. They validate implementation quality, system wiring, evaluation discipline, fallback logic, and local reproducibility. They are not production performance, production latency, or real user-impact claims.

## Future Real-Data Evaluation Plan

- Ingest a larger public dataset such as Amazon Reviews 2023.
- Rebuild retrieval, recommendation, sequence, ranking, text/metadata representation, and GenRec artifacts on larger splits.
- Add real sentence-transformer and image embeddings.
- Add true FAISS indexes and larger candidate pools.
- Run prompt and provider ablations for real LLM query understanding and explanations.
- Add human evaluation for explanation usefulness and GenRec answer quality.
- Add production-style API latency/load tests.
- Track online metrics only if the system is later deployed to real traffic.

## Reproducibility Commands

```bash
python3 src/pipelines/build_dataset.py --config configs/data/debug_sample.yaml
python3 src/pipelines/generate_queries.py --config configs/data/query_generation_debug.yaml
python3 src/pipelines/evaluate_retrieval.py --config configs/retrieval/bm25_debug.yaml
python3 src/pipelines/evaluate_recommenders.py --config configs/recommendation/itemcf_debug.yaml
python3 src/pipelines/train_sequence_model.py --config configs/sequence/sasrec_debug.yaml
python3 src/pipelines/evaluate_sequence_model.py --config configs/sequence/sasrec_debug.yaml
python3 src/pipelines/build_ranking_dataset.py --config configs/ranking/ranking_dataset_debug.yaml
python3 src/pipelines/train_ranker.py --config configs/ranking/lightgbm_ranker_debug.yaml
python3 src/pipelines/evaluate_ranker.py --config configs/ranking/lightgbm_ranker_debug.yaml
python3 src/pipelines/evaluate_llm_query_understanding.py --config configs/llm/query_understanding.yaml
python3 src/pipelines/generate_user_profiles.py --config configs/llm/user_profile.yaml
python3 src/pipelines/build_multimodal_embeddings.py --config configs/multimodal/text_metadata_fusion_debug.yaml
python3 src/pipelines/evaluate_multimodal.py --config configs/multimodal/text_metadata_fusion_debug.yaml
python3 src/pipelines/evaluate_genrec.py --config configs/genrec/genrec_debug.yaml
python3 src/pipelines/generate_validation_report.py --config validation/experiments/stage5_debug_validation.yaml
bash scripts/run_validation.sh
```
