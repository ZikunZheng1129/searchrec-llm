# Recommendation Baselines

The recommendation layer adds classic baselines on top of the local synthetic debug dataset. These models are deliberately simple and lightweight so deeper sequence, ranking, and LLM-assisted components have honest baselines.

## Training And Evaluation

All recommenders train on `data/processed/train.parquet`.

Evaluation uses the configured split, usually `data/processed/test.parquet`. For each evaluation user, the target items are that user's `item_id` values in the evaluation split. Recommendations exclude items already seen in the train split when `exclude_seen=true`.

The leave-one-out split creates one test item per user, so this is a next-item top-K recommendation setup.

## Popularity

The popularity recommender scores items by aggregate training interaction strength. The default score is `sum(event_weight)`. Unknown users receive the same global popular-item list.

Run:

```bash
python3 src/pipelines/evaluate_recommenders.py --config configs/recommendation/popularity_debug.yaml
```

## ItemCF

The item-based collaborative filtering baseline builds a user-item implicit matrix from training interactions and computes item-item cosine similarity with NumPy. A user's candidate score is the weighted sum of similarities from their seen items.

Run:

```bash
python3 src/pipelines/evaluate_recommenders.py --config configs/recommendation/itemcf_debug.yaml
```

## Matrix Factorization

The matrix factorization baseline is a small NumPy implementation trained with stochastic gradient descent. Positive pairs come from training interactions and negative pairs are sampled from items the user has not interacted with. This is a local baseline, not a production MF system.

Run:

```bash
python3 src/pipelines/evaluate_recommenders.py --config configs/recommendation/matrix_factorization_debug.yaml
```

## User-History Embedding

The user-history embedding recommender reuses local TF-IDF item vectors. A user profile is the weighted average of vectors for items in their training history, and candidate items are scored by cosine similarity.

Run:

```bash
python3 src/pipelines/evaluate_recommenders.py --config configs/recommendation/user_history_embedding_debug.yaml
```

## Metrics

Results are written to `validation/results/recommender_baselines.csv`.

Metrics:

- `HitRate@K`: 1 if any target item appears in top K, averaged over users.
- `Recall@K`: fraction of target items found in top K, averaged over users.
- `NDCG@K`: rank-discounted binary relevance, averaged over users.
- `MRR@K`: reciprocal rank of the first target item in top K, averaged over users.
- `Coverage@K`: unique recommended item IDs divided by total catalog item count.
- `avg_latency_ms` and `p95_latency_ms`: per-user recommendation latency.

Reruns replace rows for the same method, split, and config path, then sort by method and config path.

## Limitations

These baselines use synthetic data and small local matrices. They do not model long-term sequences, multimodal content, neural retrieval, or ranker features.

Future work includes SASRec, GRU4Rec, BERT4Rec, two-tower retrieval, personalized ranking, and multimodal user/item embeddings.
