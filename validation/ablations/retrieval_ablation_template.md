# Retrieval Ablation Template

## Experiment Question

What retrieval design choice are we testing?

Example questions:

- Does hybrid retrieval improve over BM25?
- Does `top_k=100` improve `Recall@50` without hurting latency?
- Does real FAISS change latency versus NumPy fallback?

## Hypothesis

State the expected outcome and why.

## Compared Methods/Configs

- Baseline:
- Variant:

## Dataset/Split

- Dataset:
- Split:

## Metrics

- Recall@K:
- MRR@K:
- Query coverage:
- Latency:

## Results

| method | config | primary metric | secondary metric | coverage | avg latency |
| --- | --- | --- | --- | --- | --- |
| TBD | TBD | TBD | TBD | TBD | TBD |

## Decision

Document the selected option and rationale.

## Error Analysis

Summarize representative wins, losses, and failure categories.

## Next Steps

List follow-up experiments or fixes.
