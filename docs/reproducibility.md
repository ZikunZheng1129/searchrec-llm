# Reproducibility

This guide lists the current repository commands by capability. Run commands from the repository root after cloning or unpacking the project. The default workflow uses synthetic debug data and deterministic mock LLM clients; no OpenAI API key or external API call is required.

## Environment Setup

Python `>=3.10` is required.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If `python3.10` is unavailable but another Python 3.10+ interpreter is installed, create the virtual environment with that interpreter.

## Build Synthetic Data

```bash
python3 src/pipelines/build_dataset.py --config configs/data/debug_sample.yaml
```

Expected outputs:

- `data/processed/train.parquet`
- `data/processed/val.parquet`
- `data/processed/test.parquet`
- `data/processed/item_metadata.parquet`
- `data/processed/user_sequences.parquet`
- `data/processed/negative_samples.parquet`

The inspected artifacts contain 50 users, 100 items, 400 train interactions, 50 validation interactions, 50 test interactions, 50 user sequences, and 2,500 negative samples.

## Generate Search Queries

```bash
python3 src/pipelines/generate_queries.py --config configs/data/query_generation_debug.yaml
```

Expected output:

- `data/processed/query_item_pairs.parquet`

The inspected artifact contains 300 query-item pairs, 94 unique query texts, and split counts of 243 train, 31 validation, and 26 test rows.

## Build And Evaluate Retrieval

Build indexes:

```bash
python3 src/pipelines/build_index.py --config configs/retrieval/bm25_debug.yaml
python3 src/pipelines/build_index.py --config configs/retrieval/dense_debug.yaml
python3 src/pipelines/build_index.py --config configs/retrieval/faiss_debug.yaml
python3 src/pipelines/build_index.py --config configs/retrieval/hybrid_retrieval.yaml
```

Evaluate retrieval:

```bash
python3 src/pipelines/evaluate_retrieval.py --config configs/retrieval/bm25_debug.yaml
python3 src/pipelines/evaluate_retrieval.py --config configs/retrieval/dense_debug.yaml
python3 src/pipelines/evaluate_retrieval.py --config configs/retrieval/faiss_debug.yaml
python3 src/pipelines/evaluate_retrieval.py --config configs/retrieval/hybrid_retrieval.yaml
```

Expected outputs:

- `data/indexes/bm25_debug.pkl`
- `data/indexes/dense_debug.pkl`
- `data/indexes/faiss_debug.pkl`
- `data/indexes/hybrid_debug.pkl`
- `validation/results/retrieval_results.csv`

In the inspected environment, FAISS is unavailable, so the FAISS-style row reports the NumPy backend.

## Evaluate Recommendation Baselines

```bash
python3 src/pipelines/evaluate_recommenders.py --config configs/recommendation/popularity_debug.yaml
python3 src/pipelines/evaluate_recommenders.py --config configs/recommendation/itemcf_debug.yaml
python3 src/pipelines/evaluate_recommenders.py --config configs/recommendation/matrix_factorization_debug.yaml
python3 src/pipelines/evaluate_recommenders.py --config configs/recommendation/user_history_embedding_debug.yaml
```

Expected output:

- `validation/results/recommender_baselines.csv`

## Train And Evaluate Sequence Models

GRU4Rec:

```bash
python3 src/pipelines/train_sequence_model.py --config configs/sequence/gru4rec_debug.yaml
python3 src/pipelines/evaluate_sequence_model.py --config configs/sequence/gru4rec_debug.yaml
```

SASRec:

```bash
python3 src/pipelines/train_sequence_model.py --config configs/sequence/sasrec_debug.yaml
python3 src/pipelines/evaluate_sequence_model.py --config configs/sequence/sasrec_debug.yaml
```

Expected outputs:

- `artifacts/models/sequence/gru4rec_debug.pt`
- `artifacts/models/sequence/sasrec_debug.pt`
- `validation/results/sequence_results.csv`

## Build Ranking Candidates

```bash
python3 src/pipelines/build_ranking_dataset.py --config configs/ranking/ranking_dataset_debug.yaml
```

Expected output:

- `data/processed/ranking_candidates.parquet`

The inspected artifact contains 15,000 rows: 300 queries with 50 candidates each, positive candidate coverage of 1.0, and no duplicate query-candidate pairs.

## Train And Evaluate Rankers

```bash
python3 src/pipelines/train_ranker.py --config configs/ranking/lightgbm_ranker_debug.yaml
python3 src/pipelines/evaluate_ranker.py --config configs/ranking/lightgbm_ranker_debug.yaml

python3 src/pipelines/train_ranker.py --config configs/ranking/mlp_ranker_debug.yaml
python3 src/pipelines/evaluate_ranker.py --config configs/ranking/mlp_ranker_debug.yaml

python3 src/pipelines/train_ranker.py --config configs/ranking/cross_encoder_ranker_debug.yaml
python3 src/pipelines/evaluate_ranker.py --config configs/ranking/cross_encoder_ranker_debug.yaml

python3 src/pipelines/train_ranker.py --config configs/ranking/mixed_ranker_debug.yaml
python3 src/pipelines/evaluate_ranker.py --config configs/ranking/mixed_ranker_debug.yaml
```

Expected outputs:

- `artifacts/models/ranking/lightgbm_ranker_debug.pkl`
- `artifacts/models/ranking/mlp_ranker_debug.pt`
- `artifacts/models/ranking/cross_encoder_ranker_debug.pt`
- `artifacts/models/ranking/mixed_ranker_debug.pkl`
- `validation/results/ranking_results.csv`

In the inspected environment, LightGBM is unavailable, so the LightGBM-compatible row reports the `numpy_linear` backend.

## Run Query Understanding And Profile Generation

Structured mock query understanding:

```bash
python3 src/pipelines/evaluate_llm_query_understanding.py --config configs/llm/query_understanding.yaml
```

Behavior-grounded mock user profiles:

```bash
python3 src/pipelines/generate_user_profiles.py --config configs/llm/user_profile.yaml
```

Expected outputs:

- `data/processed/llm_query_understanding.parquet`
- `data/processed/user_profiles.parquet`
- `data/processed/user_profile_embeddings.parquet`
- `validation/results/llm_query_understanding_results.csv`
- `validation/results/user_profile_results.csv`

The default configs use `MockLLMClient` with `allow_external_api_calls: false`.

## Build And Evaluate Text/Metadata Representations

Text-only:

```bash
python3 src/pipelines/build_multimodal_embeddings.py --config configs/multimodal/text_only_debug.yaml
python3 src/pipelines/evaluate_multimodal.py --config configs/multimodal/text_only_debug.yaml
```

Metadata-only:

```bash
python3 src/pipelines/build_multimodal_embeddings.py --config configs/multimodal/metadata_only_debug.yaml
python3 src/pipelines/evaluate_multimodal.py --config configs/multimodal/metadata_only_debug.yaml
```

Text + metadata fusion:

```bash
python3 src/pipelines/build_multimodal_embeddings.py --config configs/multimodal/text_metadata_fusion_debug.yaml
python3 src/pipelines/evaluate_multimodal.py --config configs/multimodal/text_metadata_fusion_debug.yaml
```

Representation scores for ranking candidates:

```bash
python3 src/pipelines/build_multimodal_embeddings.py --config configs/multimodal/multimodal_fusion_debug.yaml
python3 src/pipelines/evaluate_multimodal.py --config configs/multimodal/multimodal_fusion_debug.yaml
python3 src/pipelines/augment_ranking_with_multimodal.py --config configs/multimodal/multimodal_fusion_debug.yaml
```

Expected outputs:

- `data/embeddings/item_text_embeddings.parquet`
- `data/embeddings/item_metadata_embeddings.parquet`
- `data/embeddings/item_multimodal_embeddings.parquet`
- `data/processed/ranking_candidates_multimodal.parquet`
- `validation/results/multimodal_results.csv`

The current debug catalog has no real image embeddings. The inspected embedding artifacts report `image_available_rate=0.0`.

## Run GenRec Evaluation

Complete mock GenRec suite:

```bash
python3 src/pipelines/evaluate_genrec.py --config configs/genrec/genrec_debug.yaml
```

Focused experiments:

```bash
python3 src/pipelines/evaluate_genrec.py --config configs/genrec/direct_generator_debug.yaml
python3 src/pipelines/evaluate_genrec.py --config configs/genrec/constrained_generator_debug.yaml
python3 src/pipelines/evaluate_genrec.py --config configs/genrec/llm_reranker_debug.yaml
python3 src/pipelines/evaluate_genrec.py --config configs/genrec/explanation_debug.yaml
```

Expected outputs:

- `data/processed/genrec_outputs.parquet`
- `data/processed/llm_rerank_outputs.parquet`
- `data/processed/recommendation_explanations.parquet`
- `validation/results/genrec_results.csv`
- `validation/reports/final_llm_decision.md`

The default configs use the deterministic mock GenRec client. The evaluation validates parsing, catalog constraints, fallback behavior, and rule-based explanation checks; it is not a real-LLM quality benchmark.

## Generate Validation Reports

```bash
python3 src/pipelines/generate_validation_report.py --config validation/experiments/stage5_debug_validation.yaml
```

Expected outputs:

- `validation/results/final_leaderboard.csv`
- `validation/reports/final_model_selection_report.md`
- `validation/reports/latency_quality_tradeoff.md`
- `validation/reports/validation_summary.md`

The final leaderboard normalizes component-specific CSVs into one table, but metrics remain component-specific and should not be compared as one global score.

## Run The Local Demo

API:

```bash
bash scripts/launch_api.sh
```

Open `http://127.0.0.1:8000/docs`.

Dashboard:

```bash
bash scripts/launch_dashboard.sh
```

Open `http://localhost:8501`.

Both surfaces read local artifacts and use the mock LLM path by default.

## Run Tests And Package Checks

```bash
python3 -m pytest tests/
python3 -m ruff check .
python3 -m ruff format --check .
bash scripts/check_project_package.sh
```

The Makefile also provides:

```bash
make test
make lint
make api
make dashboard
make demo
```

## Bounded Validation Script

The repository includes a validation script:

```bash
bash scripts/run_validation.sh
```

By default it rebuilds the data, query generation, retrieval evaluations, recommendation evaluations, and report generation. Heavier groups are opt-in:

```bash
RUN_RANKING=1 RUN_LLM=1 RUN_MULTIMODAL=1 RUN_GENREC=1 bash scripts/run_validation.sh
```

Use the opt-in flags when you intentionally want to refresh the broader local artifacts.
