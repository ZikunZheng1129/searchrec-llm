# Demo Script

This guide provides a local technical walkthrough for the API and dashboard. The demo uses synthetic debug data and mock LLM outputs by default. No external API key is required.

## Setup

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
python3 src/pipelines/generate_validation_report.py --config validation/experiments/stage5_debug_validation.yaml
```

## Start The API

```bash
bash scripts/launch_api.sh
```

Open `http://127.0.0.1:8000/docs` and check:

```bash
curl http://127.0.0.1:8000/health
```

## Start The Dashboard

```bash
bash scripts/launch_dashboard.sh
```

Open `http://127.0.0.1:8501`.

## Suggested Query Examples

Use short catalog-style queries:

- `gift beauty`
- `affordable electronics`
- `running sports`
- `wireless audio`
- `home kitchen`

These are examples for the synthetic catalog. If a query returns sparse results, use the dashboard's sample queries or inspect `data/processed/query_item_pairs.parquet`.

## Demo Flow

1. Open the dashboard landing page and verify the synthetic/mock caveat.
2. Show artifact status to prove the demo reads local generated outputs.
3. Run a search query and explain BM25/dense/hybrid retrieval.
4. Show LLM query understanding and explain structured JSON outputs.
5. Show user profile generation and how profiles can feed personalization.
6. Show the ranking pipeline and final leaderboard.
7. Show GenRec and explain why candidate constraints prevent invalid item IDs.
8. Show model comparison and latency/quality tradeoffs.
9. Show error analysis and business/proxy metrics.

## System Summary

SearchRec-LLM is a local production-style discovery system. It starts with synthetic user/item data, builds retrieval, recommendation, sequence, ranking, LLM understanding, text/metadata representation, and GenRec layers, validates each layer with standardized metrics, and serves the result through FastAPI and Streamlit. The key design choice is candidate-constrained GenRec: the LLM can help rerank and explain, but final recommendations stay grounded in retrieved catalog candidates.

## Five-Minute Technical Walkthrough

1. Data: deterministic synthetic users, items, interactions, sequences, negatives, and query-item pairs.
2. Retrieval: BM25, TF-IDF dense, FAISS-style NumPy fallback, and hybrid recall.
3. Recommendation: popularity, itemCF, matrix factorization, and user-history embedding.
4. Sequence/ranking: GRU4Rec/SASRec, ranking features, MLP, tiny cross-encoder, and mixed ranker.
5. LLM/content representation: mock query understanding, user profiles, text/metadata fusion.
6. GenRec: direct generation baseline, constrained generation, parser, hallucination checker, fallback, explanations.
7. Validation/demo: final leaderboard, reports, API, and dashboard.

## Ten-Minute Deep Dive

Spend extra time on:

- Why unit tests and validation reports are separate.
- Why candidate-constrained GenRec is safer than direct generation.
- How fallback behavior keeps the project runnable locally.
- How metrics are selected per component and not compared globally.
- How this would scale with real logs, real embeddings, FAISS, and online evaluation.

## Limitations

The biggest limitation is that current metrics are from synthetic debug data. They validate the pipeline and model-selection framework, not production performance. Next steps would include a larger public dataset, real embeddings, true ANN retrieval, human evaluation, and API load testing.

## Mock LLM Rationale

Mock LLMs keep the default demo deterministic and safe to run without API keys. The code has provider abstractions, while the default path remains reproducible in a local environment.

## Scaling Path

Future scaling work would replace debug data with real logs or a larger public dataset, move dense retrieval to FAISS, cache candidate sets, train larger sequence/ranking models on GPU, add real image embeddings, add prompt/provider evaluation, and track latency budgets with production-style monitoring.
