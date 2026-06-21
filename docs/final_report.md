# SearchRec-LLM Final Report

## Executive Summary

SearchRec-LLM is a local, production-style system for search, recommendation, ranking, text/metadata item representation, and candidate-constrained GenRec. It demonstrates the engineering shape of a modern discovery stack while staying reproducible on synthetic debug data and deterministic mock LLM clients.

The project is not production performance evidence, not a production deployment, and does not claim platform-internal data, real user traffic, or real business impact. Its value is in system design, modular implementation, evaluation discipline, fallback behavior, and an end-to-end demo.

## Problem And Motivation

Modern discovery systems need to understand intent, retrieve relevant candidates, personalize recommendations, rank results, handle cold-start catalog items, incorporate content signals, and safely use LLMs without letting them hallucinate unavailable products. SearchRec-LLM turns those requirements into a modular project that can be inspected, tested, evaluated, and demonstrated.

## Technical Scope

This project maps to large-scale recommendation, e-commerce search, content recommendation, product retrieval/ranking, and search/recommendation infrastructure roles because it includes query understanding, candidate recall, recommender baselines, sequential models, multi-stage ranking, text/metadata item representation, GenRec safety design, validation reports, and serving surfaces.

## Dataset And Data Schema

The data layer generates deterministic synthetic users, items, interactions, user sequences, negative samples, and query-item pairs. Item metadata includes fields such as title, category, brand, and description. The synthetic nature is intentional: it keeps the project shareable and avoids private-data assumptions.

## End-to-End System Architecture

The architecture flows from synthetic data to query generation, retrieval, recommendation, sequential modeling, ranking, LLM query understanding, user profiles, text/metadata item representations, candidate-constrained GenRec, and local serving through FastAPI and Streamlit. See `docs/system_design.md` for the full module and artifact flow.

## Retrieval System

The retrieval system implements BM25, TF-IDF dense retrieval, a FAISS-style vector interface with safe NumPy fallback, and hybrid retrieval. On the current synthetic debug query set, all retrieval baselines reach Recall@50 1.000000, and BM25 is selected because it has the best MRR@10 at 0.228785.

## Recommendation Baselines

The recommendation system implements popularity, itemCF, matrix factorization, and user-history embedding recommenders. itemCF is the strongest current recommendation baseline with NDCG@10 0.055705 and coverage@10 0.980000.

## Sequential Recommendation

The sequential modeling layer includes GRU4Rec and SASRec. SASRec is selected on the current debug split with NDCG@10 0.029743, MRR@10 0.020000, and coverage@10 0.670000.

## Multi-Stage Ranking

The ranking layer builds candidates, features, and multiple rankers. The LightGBM-style wrapper using the `numpy_linear` backend is the strongest current ranker with NDCG@10 0.299915 and MRR@10 0.227442.

## LLM Query Understanding And User Profiles

The LLM-assisted query/profile layer introduces a mock-first abstraction for structured query understanding and user profile generation. The query-understanding mock reaches schema_valid_rate 1.000000 and intent_match_rate 1.000000 on synthetic labels. These results validate structured behavior, not real LLM quality.

## Text And Metadata Item Representation

The content-representation layer implements local text, metadata, optional image interface, and fusion embeddings. `text_metadata_fusion` is the strongest current representation row with NDCG@10 0.292996. The current dataset has no real image features, so image support remains an interface and future extension.

## Candidate-Constrained GenRec And Explanations

The GenRec layer compares direct generation, LLM reranking, candidate-constrained generation, and explanations. Candidate-constrained GenRec is preferred because it grounds generated item IDs in known candidates: valid_item_rate 1.000000 and hallucination_rate 0.000000, compared with direct generation valid_item_rate 0.900000 and hallucination_rate 0.100000.

## API And Dashboard Demo

The serving layer adds a FastAPI service and Streamlit dashboard. The service exposes health, artifact status, search, recommendation, query understanding, GenRec, leaderboard, business metrics, and error-taxonomy endpoints. The dashboard offers a guided demo over the same local artifacts.

## Validation Framework

The validation framework reads real result CSVs, normalizes rows into a final leaderboard, and generates model-selection and latency/quality reports. Metrics are selected within their own component only. Retrieval, recommendation, ranking, text/metadata representation, and GenRec metrics are not globally interchangeable.

## Final Results Summary

See `docs/final_results_summary.md` for exact current tables. The current selected rows are BM25 for retrieval, itemCF for recommendation, SASRec for sequence, LightGBM-style ranking with NumPy fallback for ranking, mock query understanding for structured parsing, text-metadata fusion for content retrieval, and candidate-constrained generation for GenRec.

## Design Decisions And Tradeoffs

- Local-first implementation keeps the project reproducible and reviewable.
- Synthetic data allows end-to-end validation without private logs.
- Simple baselines come before complex models, making gains and tradeoffs visible.
- LLM components are mock-first and candidate-grounded by default.
- Candidate-constrained generation prioritizes catalog validity over unconstrained creativity.
- Fallbacks keep the system runnable without optional heavy dependencies.

## Limitations

Current validation is based on a small synthetic dataset. There is no platform-internal data, no real production traffic, no online A/B testing, no human evaluation, no production-scale latency testing, and no real image embeddings. Business metrics shown in the dashboard are proxy metrics, not GMV or revenue claims.

## Future Work

Future work includes public real-data ingestion, larger Colab/GPU training, true FAISS indexes, neural text and image embeddings, two-tower retrieval, BERT4Rec, real LightGBM/XGBoost ranking, prompt/provider ablations, human evaluation, API load testing, and deployment polish.

## Reproducibility Commands

```bash
bash scripts/run_validation.sh
python3 src/pipelines/generate_validation_report.py --config validation/experiments/stage5_debug_validation.yaml
bash scripts/launch_api.sh
bash scripts/launch_dashboard.sh
```

## Technical Takeaway

SearchRec-LLM demonstrates a full local discovery system, component-level model comparison with validation artifacts, LLM integration through candidate constraints and fallbacks, and reproducible API/dashboard demos.
