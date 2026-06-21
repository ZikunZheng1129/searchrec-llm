# Resume Bullets

Use these as source material and trim for the target role. These bullets are intentionally honest: the project is local, portfolio-focused, and validated on synthetic debug data with mock LLM outputs by default.

## One-Line Version

Built SearchRec-LLM, a local search/recommendation system with hybrid retrieval, sequential recommendation, multi-stage ranking, text/metadata item representations, LLM query understanding, candidate-constrained GenRec, FastAPI serving, and a Streamlit dashboard.

## Two-Line Version

Built SearchRec-LLM, an end-to-end local discovery system covering data generation, query understanding, retrieval, recommendation, sequential modeling, ranking, text/metadata item representation, candidate-constrained GenRec, validation, FastAPI, and Streamlit.

Implemented systematic offline validation across retrieval, recommendation, ranking, text/metadata representation, and LLM/GenRec components using Recall@K, NDCG@K, MRR, coverage, latency, valid-item rate, hallucination rate, and explanation faithfulness.

## Three To Five Bullet Version

- Built a local end-to-end search and recommendation portfolio system with synthetic data generation, retrieval, recommendation, sequence models, ranking, text/metadata item representations, GenRec, API serving, and dashboard demo.
- Implemented BM25, TF-IDF dense retrieval, FAISS-style NumPy fallback, hybrid retrieval, popularity, itemCF, matrix factorization, user-history embedding, GRU4Rec, SASRec, MLP ranking, tiny cross-encoder ranking, and mixed ranking.
- Designed candidate-constrained GenRec to ground LLM-style recommendations in retrieved catalog candidates, reducing invalid item risk versus direct generation in local synthetic experiments.
- Created a validation framework that standardizes component result CSVs and compares quality, coverage, latency, valid-item rate, hallucination rate, and explanation faithfulness.
- Packaged the project with FastAPI endpoints, Streamlit demo pages, Docker files, final reports, resume bullets, and interview-ready documentation.

## ML Engineer Version

- Developed a modular ML discovery stack with deterministic data pipelines, retrievers, recommender baselines, sequential models, rankers, text/metadata encoders, mock-first LLM modules, and reproducible validation artifacts.
- Built evaluation workflows for Recall@K, NDCG@K, MRR, AUC, coverage, latency, schema validity, valid-item rate, hallucination rate, and explanation faithfulness.
- Added fallback-aware implementations, including NumPy vector search fallback, LightGBM-style ranking fallback, and mock LLM clients for reproducible local validation.

## Search And Recommendation Role Version

- Implemented a modular search/recommendation system with lexical/dense/hybrid candidate recall, collaborative and sequence recommendation, ranking, text/metadata item representation, and candidate-grounded GenRec.
- Compared retrieval and ranking methods with standardized offline metrics and selected component-level baselines through validation reports rather than arbitrary model choice.
- Demonstrated how LLM query understanding and generated recommendations can augment retrieval/ranking while remaining grounded in catalog candidates.

## LLM / GenAI Role Version

- Built mock-first LLM query understanding, user profile generation, candidate reranking, candidate-constrained GenRec, output parsing, hallucination checks, and evidence-grounded explanations.
- Evaluated GenRec designs with valid-item rate, hallucination rate, parse success, fallback rate, NDCG@10, MRR@10, explanation faithfulness, latency, and cost placeholders.
- Chose candidate-constrained GenRec over direct generation because it preserves catalog validity in local synthetic experiments.

## Backend / ML Systems Version

- Served local ML artifacts through FastAPI and Streamlit with typed schemas, health checks, artifact status, search/recommendation endpoints, GenRec endpoints, leaderboard views, and demo pages.
- Organized reproducible pipelines, YAML configs, tests, validation reports, and shell scripts for local end-to-end operation.
- Added documentation and package checks to make the repository reviewable for portfolio and interview use.
