# Recommendation Ablation Template

## Experiment Question

What recommendation design choice are we testing?

Example questions:

- Does itemCF improve over popularity?
- Does matrix factorization improve `NDCG@10`?
- Does user-history embedding improve coverage?
- Does excluding seen items affect quality?

## Hypothesis

State the expected outcome and why.

## Compared Methods/Configs

- Baseline:
- Variant:

## Dataset/Split

- Dataset:
- Split:

## Metrics

- HitRate@K:
- Recall@K:
- NDCG@K:
- MRR@K:
- Coverage:
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
