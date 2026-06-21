#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Running Stage 1 dataset build..."
python3 src/pipelines/build_dataset.py --config configs/data/debug_sample.yaml

echo "Running Stage 2 query generation..."
python3 src/pipelines/generate_queries.py --config configs/data/query_generation_debug.yaml

echo "Running Stage 3 retrieval evaluations..."
python3 src/pipelines/evaluate_retrieval.py --config configs/retrieval/bm25_debug.yaml
python3 src/pipelines/evaluate_retrieval.py --config configs/retrieval/dense_debug.yaml
python3 src/pipelines/evaluate_retrieval.py --config configs/retrieval/faiss_debug.yaml
python3 src/pipelines/evaluate_retrieval.py --config configs/retrieval/hybrid_retrieval.yaml

echo "Running Stage 4 recommendation evaluations..."
python3 src/pipelines/evaluate_recommenders.py --config configs/recommendation/popularity_debug.yaml
python3 src/pipelines/evaluate_recommenders.py --config configs/recommendation/itemcf_debug.yaml
python3 src/pipelines/evaluate_recommenders.py --config configs/recommendation/matrix_factorization_debug.yaml
python3 src/pipelines/evaluate_recommenders.py --config configs/recommendation/user_history_embedding_debug.yaml

if [[ "${RUN_RANKING:-0}" == "1" ]]; then
  echo "Running Stage 7 ranking evaluations..."
  python3 src/pipelines/build_ranking_dataset.py --config configs/ranking/ranking_dataset_debug.yaml
  python3 src/pipelines/train_ranker.py --config configs/ranking/lightgbm_ranker_debug.yaml
  python3 src/pipelines/evaluate_ranker.py --config configs/ranking/lightgbm_ranker_debug.yaml
  python3 src/pipelines/train_ranker.py --config configs/ranking/mlp_ranker_debug.yaml
  python3 src/pipelines/evaluate_ranker.py --config configs/ranking/mlp_ranker_debug.yaml
  python3 src/pipelines/train_ranker.py --config configs/ranking/cross_encoder_ranker_debug.yaml
  python3 src/pipelines/evaluate_ranker.py --config configs/ranking/cross_encoder_ranker_debug.yaml
  python3 src/pipelines/train_ranker.py --config configs/ranking/mixed_ranker_debug.yaml
  python3 src/pipelines/evaluate_ranker.py --config configs/ranking/mixed_ranker_debug.yaml
fi

if [[ "${RUN_LLM:-0}" == "1" ]]; then
  echo "Running Stage 8 mock LLM evaluations..."
  python3 src/pipelines/evaluate_llm_query_understanding.py --config configs/llm/query_understanding.yaml
  python3 src/pipelines/generate_user_profiles.py --config configs/llm/user_profile.yaml
fi

if [[ "${RUN_MULTIMODAL:-0}" == "1" ]]; then
  echo "Running Stage 9 multimodal evaluations..."
  python3 src/pipelines/build_multimodal_embeddings.py --config configs/multimodal/text_only_debug.yaml
  python3 src/pipelines/evaluate_multimodal.py --config configs/multimodal/text_only_debug.yaml
  python3 src/pipelines/build_multimodal_embeddings.py --config configs/multimodal/metadata_only_debug.yaml
  python3 src/pipelines/evaluate_multimodal.py --config configs/multimodal/metadata_only_debug.yaml
  python3 src/pipelines/build_multimodal_embeddings.py --config configs/multimodal/text_metadata_fusion_debug.yaml
  python3 src/pipelines/evaluate_multimodal.py --config configs/multimodal/text_metadata_fusion_debug.yaml
  python3 src/pipelines/build_multimodal_embeddings.py --config configs/multimodal/multimodal_fusion_debug.yaml
  python3 src/pipelines/evaluate_multimodal.py --config configs/multimodal/multimodal_fusion_debug.yaml
fi

if [[ "${RUN_GENREC:-0}" == "1" ]]; then
  echo "Running Stage 10 mock GenRec evaluations..."
  python3 src/pipelines/evaluate_genrec.py --config configs/genrec/direct_generator_debug.yaml
  python3 src/pipelines/evaluate_genrec.py --config configs/genrec/constrained_generator_debug.yaml
  python3 src/pipelines/evaluate_genrec.py --config configs/genrec/llm_reranker_debug.yaml
  python3 src/pipelines/evaluate_genrec.py --config configs/genrec/explanation_debug.yaml
  python3 src/pipelines/evaluate_genrec.py --config configs/genrec/genrec_debug.yaml
fi

echo "Generating Stage 5 validation artifacts..."
python3 src/pipelines/generate_validation_report.py --config validation/experiments/stage5_debug_validation.yaml

echo "Validation outputs:"
echo "- validation/results/final_leaderboard.csv"
echo "- validation/reports/final_model_selection_report.md"
echo "- validation/reports/latency_quality_tradeoff.md"
echo "- validation/reports/validation_summary.md"
