# Multi-Stage Ranking

The ranking layer scores query-candidate pairs from local retrieval and applies learned or deterministic reranking over synthetic query-item examples.

## Architecture

The local pipeline is:

1. Candidate recall from BM25, TF-IDF dense retrieval, and hybrid retrieval.
2. Ranking feature generation for each query-candidate pair.
3. Feature ranker with optional LightGBM and a deterministic NumPy fallback.
4. PyTorch MLP ranker over numeric features.
5. Tiny local cross-encoder reranker over query and item text.
6. Mixed ranker combining relevance, authority, conversion, diversity, and
   cold-start proxies.
7. Ranking evaluation and validation leaderboard integration.

## Candidate Dataset

`data/processed/ranking_candidates.parquet` contains one row per query-candidate
pair. Each query has one known positive target item from
`query_item_pairs.parquet`; candidates are retrieved locally and the positive
target is included when configured.

Important columns include:

- `query_id`, `query_text`, `split`
- `target_item_id`, `candidate_item_id`, `label`
- `bm25_score`, `dense_score`, `hybrid_score`
- `category_match`, `brand_match`, token overlap features
- item metadata and synthetic business proxy features

## Feature Groups

- Retrieval features: sparse, dense, and hybrid scores/ranks.
- Lexical features: title and description token overlap, category and brand
  match.
- Metadata features: category, brand, price, rating count, average rating.
- Proxy features: synthetic interaction popularity, add-to-cart count, purchase
  count, conversion proxy, authority score.
- Diversity and cold-start features: category diversity group and cold-start
  score.

Business features are synthetic proxies from the debug data. They are useful for
local modeling mechanics, not GMV or production-business claims.

## Rankers

`LightGBMRanker` tries to use LightGBM when installed. If it is unavailable, it
falls back to `NumpyLinearRanker` and reports `backend=numpy_linear`.

`MLPRankerWrapper` trains a small CPU PyTorch MLP on numeric ranking features.

`CrossEncoderRankerWrapper` trains a tiny local TransformerEncoder on
`[CLS] query [SEP] item_text` token pairs. It does not use HuggingFace,
sentence-transformers, or external embeddings.

`MixedRanker` is deterministic and blends relevance, authority, conversion,
diversity, and cold-start components with configurable weights.

## Metrics

Ranking evaluation reports:

- NDCG@10 and NDCG@20
- MRR@10 and MRR@20
- Precision@10 and Precision@20
- Recall@10 and Recall@20
- AUC
- candidate coverage
- average and p95 latency

Candidate coverage is the fraction of queries where the known positive target
appears in the candidate set.

## Run

Build ranking candidates:

```bash
python src/pipelines/build_ranking_dataset.py --config configs/ranking/ranking_dataset_debug.yaml
```

Train and evaluate rankers:

```bash
python src/pipelines/train_ranker.py --config configs/ranking/lightgbm_ranker_debug.yaml
python src/pipelines/evaluate_ranker.py --config configs/ranking/lightgbm_ranker_debug.yaml

python src/pipelines/train_ranker.py --config configs/ranking/mlp_ranker_debug.yaml
python src/pipelines/evaluate_ranker.py --config configs/ranking/mlp_ranker_debug.yaml

python src/pipelines/train_ranker.py --config configs/ranking/cross_encoder_ranker_debug.yaml
python src/pipelines/evaluate_ranker.py --config configs/ranking/cross_encoder_ranker_debug.yaml

python src/pipelines/train_ranker.py --config configs/ranking/mixed_ranker_debug.yaml
python src/pipelines/evaluate_ranker.py --config configs/ranking/mixed_ranker_debug.yaml
```

Regenerate validation reports:

```bash
python src/pipelines/generate_validation_report.py --config validation/experiments/stage5_debug_validation.yaml
```

## Caveats

- The current data is synthetic and local.
- Each query has one known positive item.
- There are no real search logs yet.
- Query ranking uses synthetic query-item candidates and does not include real query-user context.
- Business proxy features are synthetic, not real GMV or conversion signals.

## Future Work

- Real LightGBM or XGBoost ranking experiments.
- Pairwise and listwise ranking losses.
- Learned diversity.
- SASRec or user-profile ranking features when query-user context exists.
- Query-user personalized ranking.
- Production latency testing.
