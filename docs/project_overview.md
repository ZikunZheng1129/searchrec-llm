# SearchRec-LLM

A production-style local search and recommendation platform inspired by common large-scale technology and e-commerce discovery architectures.

## What This Project Is

SearchRec-LLM is a local portfolio project for large-scale recommendation, e-commerce search, product discovery, and content recommendation concepts. It demonstrates how a search and recommendation stack can move from data generation to retrieval, recommendation, ranking, LLM-assisted understanding, text/metadata item representation, candidate-constrained generation, validation, and a demo service.

The project is intentionally local-first. It uses synthetic debug data and deterministic mock LLM outputs by default, so it can be reproduced without private data, external API keys, or long-running infrastructure.

## Problem Motivation

Discovery systems have to interpret user intent, retrieve broad candidate sets, rank them for relevance and personalization, handle cold-start items, explain results, and avoid unsafe generative behavior. This project turns those concerns into a modular system where each layer can be evaluated separately.

## Target Role Alignment

- Search relevance: BM25, TF-IDF dense retrieval, hybrid recall, and ranking metrics.
- Recommendation systems: popularity, itemCF, matrix factorization, user-history embeddings, GRU4Rec, and SASRec.
- Query understanding: rule-based parsing and mock LLM structured query understanding.
- Recall and ranking: retrieval, candidate generation, ranking features, MLP, cross-encoder, and mixed ranker.
- Content representation: text, metadata, optional image interface, and fusion encoders.
- Personalization: user sequences, user-history embeddings, sequence models, and generated user profiles.
- Cold start: metadata/text retrieval and cold-start metrics.
- LLM/GenRec: direct generation baseline, candidate-constrained GenRec, reranking, parsing, hallucination checks, and explanations.
- Evaluation: standardized result CSVs, final leaderboard, latency/quality reports, and validation docs.
- Serving/demo: FastAPI service, Streamlit dashboard, Docker files, demo scripts, and package checks.

## Implemented Components

- Synthetic user, item, interaction, sequence, negative-sampling, and query-pair data pipeline.
- Retrieval baselines: BM25, TF-IDF dense retrieval, FAISS-style interface with NumPy fallback, and hybrid retrieval.
- Recommendation baselines: popularity, itemCF, matrix factorization, and user-history embedding.
- Sequential recommendation: local GRU4Rec and SASRec debug training/evaluation.
- Multi-stage ranking: feature ranker, LightGBM-style wrapper with fallback, MLP ranker, tiny cross-encoder, and mixed ranker.
- LLM query understanding and user profiles through mock-first client abstractions.
- Text/metadata item representation with an optional image-feature interface and fusion.
- Candidate-constrained GenRec with parser validation, fallback behavior, reranking, and evidence-grounded explanations.
- Validation framework with component-specific result CSVs and a final leaderboard.
- FastAPI and Streamlit dashboard for local demos.
- Final portfolio, resume, and interview documentation.

## What Is Not Claimed

- This is not production performance evidence.
- It does not use platform-internal data or architecture.
- It does not claim real GMV, revenue, engagement, or retention impact.
- It does not run online A/B tests or production traffic.
- It does not require real external LLM calls by default.
- It does not hide that current metrics come from a synthetic local debug dataset.

## Synthetic-Data Caveat

All current numeric results are from deterministic synthetic debug data. They are useful for validating system mechanics, metric wiring, comparison logic, fallback behavior, and demo readiness. They should not be interpreted as real-world model quality.

## Quick Demo Commands

```bash
python3 src/pipelines/generate_validation_report.py --config validation/experiments/stage5_debug_validation.yaml
bash scripts/launch_api.sh
bash scripts/launch_dashboard.sh
```

FastAPI docs open at `http://127.0.0.1:8000/docs`. The Streamlit dashboard opens at `http://127.0.0.1:8501` by default.
