# Interview Talking Points

## 30-Second Summary

SearchRec-LLM is my local production-style discovery system. It covers synthetic data generation, retrieval, recommendation, sequential models, ranking, LLM query understanding, user profiles, text/metadata item representation, candidate-constrained GenRec, validation reports, FastAPI, and a Streamlit dashboard. The project is synthetic and mock-first by default, so it is reproducible and honest rather than a production claim.

## Two-Minute Summary

I built SearchRec-LLM to show the architecture of a modern search and recommendation system. It starts with deterministic synthetic user/item/interactions data, generates query-item pairs, retrieves candidates with BM25/dense/hybrid methods, evaluates classic and sequential recommenders, reranks with feature and neural rankers, adds mock LLM query understanding and user profiles, builds text/metadata representations, and uses candidate-constrained GenRec to avoid hallucinated catalog items. Each component writes result CSVs and the final leaderboard selects the best method per component with quality, coverage, and latency in mind.

## Deep Technical Walkthrough

- Data: deterministic synthetic catalog, interactions, splits, sequences, negatives, and query-item pairs.
- Retrieval: BM25, TF-IDF dense retrieval, FAISS-style interface with NumPy fallback, and hybrid scoring.
- Recommendation: popularity, itemCF, matrix factorization, and user-history embedding.
- Sequential: GRU4Rec and SASRec trained/evaluated locally.
- Ranking: candidate dataset, feature ranker, LightGBM-style fallback, MLP, tiny cross-encoder, and mixed ranker.
- LLM: mock-first query understanding, user profiles, candidate reranking, constrained generation, parser, validator, and explanations.
- Content representation: text, metadata, optional image interface, and fusion encoder.
- Validation: final leaderboard and reports from real result CSVs.
- Serving: FastAPI and Streamlit demo over local artifacts.

## Why It Maps To Search, Recommendation, And E-Commerce Discovery

The project touches common system concerns in large-scale search and recommendation platforms: query intent, candidate recall, ranking, sequential behavior, cold-start content, item representation, generative recommendation safety, evaluation, and serving. It uses synthetic examples and does not depend on proprietary platform data or confidential implementation details.

## Important Design Decisions

- Local-first: reviewers can run it without accounts, API keys, or private data.
- Modular development: each component has configs, tests, artifacts, and docs.
- Validation-first: model choices come from result CSVs and reports.
- Candidate-constrained GenRec: generated recommendations must map to known item candidates.
- LLM as augmentation: retrieval and ranking remain the grounding layer.
- Fallback logic: optional dependencies and external providers are not required for tests.

## Biggest Technical Challenges

- Keeping a broad system modular without turning it into a monolith.
- Designing fair component-specific metrics while avoiding misleading global comparisons.
- Making LLM-style features deterministic and testable through mock clients.
- Grounding generated recommendations in catalog candidates.
- Keeping the demo useful without real data or external services.

## What I Learned

- Baseline discipline matters: simple baselines often reveal system issues faster than complex models.
- LLM integration needs grounding, parsing, validation, and fallback behavior.
- Tests and validation solve different problems.
- A portfolio project is stronger when limitations are explicit.

## What I Would Improve With More Time

- Replace synthetic data with Amazon Reviews 2023 or another public dataset.
- Add real neural embeddings and true FAISS indexes.
- Train larger sequence/ranking models on GPU.
- Add real image embeddings.
- Run prompt/provider ablations for real LLMs.
- Add human evaluation and API load tests.

## How I Would Scale It

Use offline batch pipelines for embeddings and features, ANN indexes for retrieval, candidate caches, feature stores for ranking, asynchronous LLM calls with budget controls, monitoring for latency and invalid outputs, and online experiments for final decisions.

## How I Would Adapt It To Real Platform Data

I would add real query logs, impressions, clicks, watches, purchases, skips, dwell time, session context, item metadata, media embeddings, and user privacy controls. Evaluation would use time-based splits, user/item cold-start slices, and online metrics.

## How I Would Evaluate Online

Start with guardrail metrics and small traffic. Track search relevance, click-through, conversion proxies, long-tail exposure, invalid recommendation rate, latency, cost, and user feedback. GenRec explanations would need human review before broad exposure.

## Answer: Why Not Just Use An LLM?

An LLM alone does not know the live catalog, availability, ranking constraints, latency budget, or personalization context. In this system, retrieval and ranking ground the candidates, and the LLM helps understand, rerank, generate constrained outputs, and explain.

## Answer: Why Synthetic Data?

Synthetic data makes the project reproducible, safe to share, and runnable without private logs. It is enough to validate architecture and mechanics. Real performance would require a larger public or production dataset.

## Answer: What Is Production-Style About This?

The project has component configs, tests, validation reports, fallback behavior, typed API schemas, dashboard demo, artifact checks, and explicit limitations. It is production-style engineering, not a production system.
