# Model Design

This document summarizes each modeling layer, its purpose, inputs, outputs, validation signal, and current limitation. All current results are from synthetic debug data and are not production performance claims.

## Retrieval

| Component | Purpose | Input | Output | Validation | Limitation |
| --- | --- | --- | --- | --- | --- |
| BM25 | Strong lexical recall baseline. | Query text and item title/category/brand/description. | Ranked item IDs with BM25 scores. | Recall@K, MRR@K, coverage, latency. | Lexical matching can miss semantic paraphrases. |
| TF-IDF dense | Lightweight local semantic-ish baseline without neural dependencies. | Tokenized item/query text. | Cosine-similarity ranked items. | Recall@K, MRR@K, coverage, latency. | Bag-of-words vectors are not true neural embeddings. |
| FAISS-style wrapper | Production-shaped vector search interface. | Dense vectors. | Nearest-neighbor ranked items. | Same retrieval metrics plus backend tracking. | Current default uses NumPy fallback when FAISS is unavailable. |
| Hybrid retrieval | Combine sparse and dense evidence. | BM25 and dense scores. | Weighted ranked candidates. | Recall@K, MRR@K, coverage, latency. | Needs weight tuning on real data. |

## Recommendation

| Component | Purpose | Input | Output | Validation | Limitation |
| --- | --- | --- | --- | --- | --- |
| Popularity | Simple global baseline. | Train interactions. | Popular item list. | HitRate@K, Recall@K, NDCG@K, MRR@K, coverage, latency. | Low personalization and low long-tail exposure. |
| itemCF | Collaborative similarity baseline. | User-item interactions. | Similar-item recommendations. | Recommendation metrics and coverage. | Sensitive to sparse histories. |
| Matrix factorization | Latent-factor collaborative baseline. | User-item interactions. | Personalized scores. | Recommendation metrics and latency. | Debug training is small and synthetic. |
| User-history embedding | Content/profile-like baseline. | User histories and item text/metadata vectors. | Personalized nearest items. | Recommendation metrics and coverage. | Limited by synthetic histories. |

## Sequential Recommendation

| Component | Purpose | Input | Output | Validation | Limitation |
| --- | --- | --- | --- | --- | --- |
| GRU4Rec | Sequential next-item baseline. | User item sequences. | Next-item scores. | HitRate/NDCG/MRR/coverage/latency. | Small debug data underuses model capacity. |
| SASRec | Attention-based sequential recommender. | User item sequences. | Next-item scores. | HitRate/NDCG/MRR/coverage/latency. | Current model is intentionally small for laptop runs. |

## Ranking

| Component | Purpose | Input | Output | Validation | Limitation |
| --- | --- | --- | --- | --- | --- |
| Feature ranker / LightGBM wrapper | Strong tabular ranking baseline with fallback. | Candidate features. | Reranked candidates. | NDCG, MRR, Precision, Recall, AUC, candidate coverage, latency. | Current backend can use a local NumPy fallback. |
| MLP ranker | Neural feature ranker. | Candidate features. | Reranked candidates. | Ranking metrics. | Small data makes deep gains unreliable. |
| Tiny local cross-encoder | Query-item interaction scorer. | Query/item text pairs. | Pairwise scores. | Ranking metrics and latency. | Debug transformer is tiny and local. |
| Mixed ranker | Blend multiple ranking signals. | Candidate scores/features. | Combined rank. | Ranking metrics and latency. | Requires calibration on real logs. |

## LLM Query Understanding

| Component | Purpose | Input | Output | Validation | Limitation |
| --- | --- | --- | --- | --- | --- |
| Structured JSON parser | Make query understanding machine-readable. | Query text. | Intent/category/brand/constraints/use cases. | Parse success, schema validity, field match rates. | Synthetic labels are simpler than real queries. |
| Mock client | Deterministic local LLM substitute. | Prompt and query. | Structured mock response. | Same LLM metrics plus zero estimated cost. | Does not measure real model quality. |
| Optional OpenAI/HF wrappers | Future real model integration points. | Prompt and query. | Provider response. | Future latency, cost, quality, and safety metrics. | Not used by default and not required for tests. |

## User Profiles

User profile generation creates evidence-based behavior summaries and profile embeddings from user histories. These profiles support personalization demos and future ranking features. Current metrics validate schema success, non-empty profiles, embedding coverage, latency, and cost on mock local outputs.

## Text And Metadata Item Representation

| Component | Purpose | Input | Output | Validation | Limitation |
| --- | --- | --- | --- | --- | --- |
| Text encoder | Represent titles/descriptions. | Item text. | Text embeddings. | Recall/NDCG/MRR, coverage, latency. | Local vectorization only. |
| Metadata encoder | Represent category, brand, price, and attributes. | Item metadata. | Metadata vectors. | Same metrics plus cold-start and coverage slices. | Metadata quality drives performance. |
| Optional image interface | Reserve path for image features. | Future image embeddings. | Image vector slot. | Future content and cold-start metrics. | Current debug data has no real image features. |
| Fusion encoder | Combine text, metadata, and optional image signals. | Available embeddings. | Fused item vectors. | Text/metadata retrieval metrics. | Needs real image embeddings later. |

## GenRec

| Component | Purpose | Input | Output | Validation | Limitation |
| --- | --- | --- | --- | --- | --- |
| Direct generation baseline | Show unconstrained generation risk. | Query/user context. | Generated item IDs. | Valid item rate, hallucination rate, parse success, NDCG, MRR. | Can produce invalid catalog items. |
| Candidate-constrained generation | Ground LLM output in retrieved/ranked candidates. | Query/user context plus candidates. | Validated item IDs. | Valid item rate, hallucination rate, fallback rate, ranking metrics. | Quality depends on candidate recall. |
| LLM reranking | Let LLM reorder a candidate set. | Candidate list and prompt. | Reranked candidates. | NDCG/MRR/Recall plus validity. | Mock by default. |
| Output parser | Convert generated text to structured IDs. | LLM text output. | Parsed item IDs and metadata. | Parse success and schema validity. | Needs robust real-world prompt tests later. |
| Hallucination checker | Detect item IDs outside the candidate/catalog set. | Parsed IDs and candidate/catalog IDs. | Validity flags. | Valid item rate and hallucination rate. | Catalog grounding only checks item identity. |
| Explanation generator | Produce evidence-grounded explanations. | Recommended items and evidence. | Template or mock LLM explanations. | Explanation faithfulness and latency. | No human preference evaluation yet. |

## API And Dashboard

FastAPI exposes local artifacts through typed endpoints. Streamlit provides a guided demo. These components validate system integration and demo readiness, not production load or production availability.
