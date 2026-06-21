# Sequential Recommendation

The sequential recommendation layer adds PyTorch next-item models over chronological user histories.

## Why Sequential Models Matter

Short-session and e-commerce recommendation can be strongly sequence-driven. The next item a user may engage with often depends on recent behavior, not only global popularity or static item similarity. GRU4Rec and SASRec provide compact baselines for modeling chronological user histories before ranking and content-representation features are added.

## Data Construction

The sequence pipeline uses processed train, validation, and test interactions.

- Item index `0` is reserved for padding.
- Real item IDs start at integer index `1`.
- Training examples are prefix-to-next-item pairs.
- Evaluation examples use the full train history as input and the validation or test item as the target.
- Sequences are left-padded so recent events appear at the end of the tensor.

Example:

```text
[i1, i2, i3] -> input [i1], target i2
[i1, i2, i3] -> input [i1, i2], target i3
```

## GRU4Rec

GRU4Rec embeds item IDs, passes the sequence through a GRU, and scores candidate items by dot product with item embeddings.

Debug run:

```bash
python3 src/pipelines/train_sequence_model.py --config configs/sequence/gru4rec_debug.yaml
python3 src/pipelines/evaluate_sequence_model.py --config configs/sequence/gru4rec_debug.yaml
```

## SASRec

SASRec uses item embeddings, positional embeddings, a causal Transformer encoder, and dot-product item scoring.

Debug run:

```bash
python3 src/pipelines/train_sequence_model.py --config configs/sequence/sasrec_debug.yaml
python3 src/pipelines/evaluate_sequence_model.py --config configs/sequence/sasrec_debug.yaml
```

## Training

The current trainer uses sampled binary cross-entropy:

- Positive items come from observed next-item targets.
- Negative items are sampled from item IDs.
- Debug configs train on CPU for two epochs.
- Larger configs are provided for later Colab or GPU runs.

## Evaluation

Metrics:

- HitRate@10/20
- Recall@10/20
- NDCG@10/20
- MRR@10/20
- Coverage@10/20
- Average and p95 latency

Results are written to `validation/results/sequence_results.csv`. Report generation includes sequence rows when this file exists and ignores it when missing.

## Local Debug Versus Larger Configs

Use debug configs locally:

- `configs/sequence/gru4rec_debug.yaml`
- `configs/sequence/sasrec_debug.yaml`

Use larger configs later on GPU:

- `configs/sequence/gru4rec.yaml`
- `configs/sequence/sasrec.yaml`

## Limitations

- Synthetic data only.
- Tiny item catalog.
- No hyperparameter tuning yet.
- No BERT4Rec yet.
- No production-scale training yet.

## Future Work

- BERT4Rec.
- Hard negatives.
- Longer sequences.
- Two-tower retrieval.
- Ranker feature integration.
