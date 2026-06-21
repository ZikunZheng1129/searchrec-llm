# GenRec Design

## Purpose

The GenRec layer adds local experiments for recommendation generation, LLM
reranking, and explanations. The important design choice is that the preferred
system is candidate-constrained: the LLM can only choose from retrieved or
ranked catalog candidates.

## Why Direct Generation Is Risky

Uncontrolled direct generation can invent item IDs, mention products outside
the catalog, or produce output that cannot be safely served. This project keeps
direct generation only as a baseline so hallucination and invalid item risk can
be measured.

## Candidate-Constrained Architecture

The preferred GenRec path is:

1. Build or load ranked candidates from the ranking and text/metadata representation workflows.
2. Format the top candidates with item ID, title, category, brand, rating,
   price, score, rank, and evidence text.
3. Prompt the LLM to return structured JSON.
4. Parse and validate the JSON.
5. Reject invalid item IDs and fall back to the original ranked candidates.
6. Evaluate validity, hallucination risk, ranking quality, latency, and cost.

## Input Sources

- `data/processed/ranking_candidates_multimodal.parquet` when available.
- `data/processed/ranking_candidates.parquet` otherwise.
- `data/processed/query_item_pairs.parquet`.
- `data/processed/item_metadata.parquet`.
- `data/processed/llm_query_understanding.parquet` when available.
- `data/processed/user_profiles.parquet` only when query rows contain real
  user IDs.

Current synthetic query examples do not provide real user-query logs, so
user-profile-conditioned GenRec is intentionally limited.

## Components

- Direct generator: unsafe baseline without candidates.
- Candidate-constrained generator: chooses only candidate item IDs.
- LLM reranker: reranks top 10 or top 20 candidate IDs.
- Output parser: handles malformed/fenced JSON and duplicate IDs.
- Hallucination checker: validates item IDs and simple explanation grounding.
- Fallback logic: returns ranked candidates when constrained output is invalid.

## Metrics

- valid item rate
- hallucination rate
- output parse success rate
- schema valid rate
- fallback rate
- Recall@10, NDCG@10, MRR@10
- explanation faithfulness
- latency and estimated cost

## Run

```bash
python3 src/pipelines/evaluate_genrec.py --config configs/genrec/genrec_debug.yaml
python3 src/pipelines/generate_validation_report.py --config validation/experiments/stage5_debug_validation.yaml
```

## Caveats

The default configs use a deterministic mock client on synthetic debug data.
The metrics validate mechanics and safety checks, not production LLM quality.
No API key is required by default.

## Future Work

- Real LLM ablations.
- User-conditioned GenRec when query-user logs exist.
- Human preference evaluation.
- Online/offline consistency checks.
- API and dashboard integration.
