# Limitations And Future Work

## Current Limitations

- Synthetic debug data only.
- No real platform logs, user traffic, proprietary architecture, or product catalog.
- No real Amazon Reviews 2023 processing yet.
- No real image embeddings yet.
- Mock LLM clients are used by default.
- No human evaluation of recommendations or explanations.
- No online A/B testing.
- No production-scale latency or load testing.
- Query-user personalization is limited because synthetic queries do not include real user context.
- Business metrics are local proxy metrics, not GMV, revenue, or engagement impact.
- Current results demonstrate system mechanics and validation discipline, not production performance.

## Future Work

- Add Amazon Reviews 2023 ingestion and larger public-data splits.
- Train larger GRU4Rec/SASRec/BERT4Rec models on Colab or GPU hardware.
- Add real sentence-transformer embeddings.
- Build real FAISS indexes for larger candidate pools.
- Add real LightGBM/XGBoost ranking backends.
- Add two-tower neural retrieval.
- Add CLIP or precomputed image embeddings.
- Run real LLM prompt, provider, cost, and latency ablations.
- Add human evaluation for explanations and GenRec answer quality.
- Add API performance tests and load-test reports.
- Add Docker polish, deployment docs, and monitoring/observability hooks.

## Productionization Plan

1. Replace synthetic data with a real public dataset and strict train/validation/test split.
2. Add offline data-quality checks, privacy-safe logging assumptions, and dataset cards.
3. Scale retrieval with approximate nearest-neighbor indexes and candidate-cache strategies.
4. Calibrate ranker features and evaluate online-safe slices.
5. Add LLM safety policies, prompt versioning, budget controls, and response auditing.
6. Run human evaluation before relying on generated explanations.
7. Add latency budgets and dashboarded operational metrics.

## Recommended Framing

This is a production-style local system, not a production system. It demonstrates one way to structure data, retrieval, recommendation, ranking, LLM augmentation, validation, and serving. The current metrics are from synthetic debug data, so they validate the pipeline and comparison framework rather than real-world performance.
