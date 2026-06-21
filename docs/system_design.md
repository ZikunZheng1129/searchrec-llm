# System Design

SearchRec-LLM is organized as a modular, local-first search and recommendation system. Each component writes explicit artifacts that can be tested, evaluated, and served by downstream components.

## End-to-End Architecture

```text
Synthetic User/Item/Interaction Data
        ↓
Query Generation + Rule-Based Query Understanding
        ↓
Hybrid Candidate Recall
BM25 + Dense TF-IDF + FAISS-style NumPy fallback
        ↓
Recommendation Baselines
Popularity + ItemCF + Matrix Factorization + User-History Embedding
        ↓
Sequential Recommendation
GRU4Rec + SASRec
        ↓
Multi-Stage Ranking
Feature Ranker + MLP + Tiny Cross-Encoder + Mixed Ranker
        ↓
LLM Query Understanding + User Profiles
Mock/OpenAI/HF client abstraction, mock by default
        ↓
Text/Metadata Item Representation
Text + Metadata + Optional Image Interface
        ↓
Candidate-Constrained GenRec
LLM rerank/generation + validation + fallback
        ↓
FastAPI + Streamlit Demo
```

## Major Modules

- `configs/`: component-specific YAML configs for data, retrieval, recommendation, sequence models, ranking, LLM, text/metadata representation, GenRec, API, and dashboard.
- `src/data/`: synthetic dataset build, splitting, negative sampling, and query generation.
- `src/retrieval/`: BM25, TF-IDF dense retrieval, FAISS-style wrapper, and hybrid retrieval.
- `src/recommendation/`: classic recommender baselines.
- `src/sequence_models/`: GRU4Rec, SASRec, sequence datasets, trainer, and evaluator.
- `src/ranking/`: ranking datasets, features, rankers, and model IO.
- `src/llm/`: mock-first LLM clients, query understanding, user modeling, prompts, GenRec, reranking, output parsing, and explanations.
- `src/multimodal/`: text, metadata, optional image-feature interface, and fusion encoders.
- `src/evaluation/`: component-specific metrics, final report generation, and validation artifacts.
- `src/pipelines/`: command-line entry points for reproducible local workflows.
- `app/api/`: FastAPI service layer.
- `app/dashboard/`: Streamlit dashboard and demo pages.
- `validation/`: experiment templates, result CSVs, reports, ablations, and error-analysis templates.
- `docs/`: technical docs and final portfolio package.

## Data Flow

The project starts with a deterministic synthetic catalog, user interactions, and train/validation/test splits. Query-item pairs are generated from the item metadata and used for search, ranking, text/metadata retrieval, and GenRec validation. User sequences support classic recommendation and sequential next-item prediction.

## Artifact Flow

Pipeline outputs are written to stable paths such as `data/processed/`, `data/indexes/`, `data/embeddings/`, `artifacts/models/`, and `validation/results/`. The validation framework reads existing result CSVs and generates `validation/results/final_leaderboard.csv` plus Markdown reports.

## Serving Flow

The FastAPI service reads local artifacts and exposes health, search, recommendation, query-understanding, GenRec, leaderboard, business-metric, and error-taxonomy endpoints. The Streamlit dashboard calls the same local artifacts and provides an interview-friendly walkthrough across search, LLM understanding, ranking, GenRec, validation, and proxy business metrics.

## Local-First Design

The default path uses synthetic data, NumPy/Pandas, local PyTorch debug models, optional FAISS fallback, and deterministic mock LLM clients. No API key is required by default. This makes the project runnable on a laptop and suitable for reproducible portfolio review.

## Colab/GPU Path

Colab or GPU hardware is useful later for larger sequence models, two-tower retrieval, BERT4Rec, cross-encoder training, real embedding generation, and image encoders. The current local demo does not require Colab.

## Why LLM Is An Augmentation Layer

The LLM layer does not replace retrieval or ranking. Retrieval and ranking produce grounded catalog candidates; LLM components help parse intent, enrich profiles, rerank candidates, produce constrained recommendations, and explain decisions. Candidate constraints keep generative output tied to known item IDs.

## Failure And Fallback Strategy

- FAISS is optional; NumPy exact search is the safe fallback.
- LightGBM-style ranking can fall back to local NumPy behavior.
- LLM clients default to deterministic mock clients.
- GenRec outputs are parsed, schema-checked, catalog-validated, and backed by fallback candidates.
- Reports clearly label synthetic debug data and mock LLM results.
