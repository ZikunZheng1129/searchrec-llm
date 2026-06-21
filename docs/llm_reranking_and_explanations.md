# LLM Reranking And Explanations

## LLM Reranking

GenRec includes LLM reranking as a candidate-constrained experiment. Unlike
the local cross-encoder, which scores query-item text pairs directly, the LLM
reranker receives a small ranked candidate list and returns structured JSON.

The reranker must only output candidate item IDs. If it emits invalid IDs or
malformed JSON, the parser marks the problem and falls back to the original
ranked candidate order.

## Evidence Selection

Explanation evidence comes only from item metadata and candidate rows:

- item ID
- title
- category
- brand
- price
- average rating
- rating count
- ranking or retrieval score when available

No product attributes are invented.

## Explanation Methods

Template explanations are deterministic and require no LLM call.

Evidence-grounded LLM explanations use the mock client by default, parse
structured JSON, and fall back to template explanations when parsing or
faithfulness checks fail.

## Faithfulness Checks

The rule-based checker verifies:

- explanation item IDs are recommended item IDs
- evidence item IDs exist in the selected evidence
- explanations do not mention categories or brands outside the evidence set

This is intentionally simple. It is a safety check for local validation, not a
complete semantic hallucination detector.

## Run

```bash
python3 src/pipelines/evaluate_genrec.py --config configs/genrec/llm_reranker_debug.yaml
python3 src/pipelines/evaluate_genrec.py --config configs/genrec/explanation_debug.yaml
```

## Future Work

- richer evidence retrieval
- citation-style explanations
- human preference evaluation
- user-facing demo
