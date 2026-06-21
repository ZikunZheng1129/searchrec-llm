# Retrieval Baselines

The retrieval layer adds local-first search baselines and evaluates them on synthetic query-item pairs.

## Item Text

Each item is indexed with searchable text built from:

- `title`
- `category`
- `brand`
- `description`

Text is lowercased, whitespace is normalized, and tokens are extracted with a simple regex. No external NLP libraries are used.

## BM25 Retriever

The BM25 retriever is a sparse lexical baseline implemented locally. It computes document frequencies, BM25 IDF values, and query-document scores over item metadata text.

Index config:

```bash
python3 src/pipelines/build_index.py --config configs/retrieval/bm25_debug.yaml
```

## Lightweight Dense TF-IDF Retriever

The dense baseline uses a custom TF-IDF vectorizer implemented with NumPy. Item and query vectors are L2-normalized and searched with cosine similarity. This is not a neural embedding model.

Index config:

```bash
python3 src/pipelines/build_index.py --config configs/retrieval/dense_debug.yaml
```

## FAISS-Style Wrapper

The FAISS-style retriever uses the TF-IDF vectors. It tries to use FAISS when available and falls back to exact NumPy similarity search when FAISS is not installed. Tests do not require FAISS.

Index config:

```bash
python3 src/pipelines/build_index.py --config configs/retrieval/faiss_debug.yaml
```

## Hybrid Retriever

The hybrid retriever combines BM25 and TF-IDF dense scores with weighted score fusion. The current config uses min-max normalization per query and equal weights.

Index config:

```bash
python3 src/pipelines/build_index.py --config configs/retrieval/hybrid_retrieval.yaml
```

## Evaluation Metrics

Retrieval is evaluated against `data/processed/query_item_pairs.parquet`.

Metrics:

- `Recall@K`: whether the target item appears in the top K retrieved items.
- `MRR@K`: reciprocal rank of the target item when it appears in the top K.
- `query_coverage`: fraction of queries that returned at least one result.
- `avg_latency_ms`: average per-query retrieval latency.
- `p95_latency_ms`: 95th percentile per-query retrieval latency.

Run evaluation:

```bash
python3 src/pipelines/evaluate_retrieval.py --config configs/retrieval/hybrid_retrieval.yaml
```

Results are written to `validation/results/retrieval_results.csv`. Reruns replace the row for the same method, split, and config path, then keep the CSV deterministic.

## Limitations

These baselines are intentionally small and local. They use synthetic query-item pairs and metadata-only item text. They do not use neural embeddings, multimodal features, personalization, or real search logs.

Future work includes sentence-transformer embeddings, real FAISS indexes, multimodal embeddings, personalized retrieval, and larger candidate pools.
