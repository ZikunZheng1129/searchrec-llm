# Final LLM Decision Report

Stage 10 status: local mock GenRec, candidate-constrained reranking/generation,
and evidence-grounded explanations are implemented.

Data caveat: current measurements use synthetic debug data and the deterministic
mock LLM client by default. These metrics validate mechanics and safety checks,
not production LLM quality.

## LLM Components Evaluated

- Stage 8 query understanding.
- Stage 8 user profile generation when user context exists.
- Stage 10 direct generation baseline.
- Stage 10 candidate-constrained generation.
- Stage 10 LLM reranking.
- Stage 10 template and evidence-grounded explanations.

## Metrics

| method | provider | model | num_queries | valid_item_rate | hallucination_rate | output_parse_success_rate | fallback_rate | ndcg_at_10 | mrr_at_10 | explanation_faithfulness | avg_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_constrained_generation | mock | mock-genrec-v1 | 26 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.2454 | 0.1674 | nan | 0.0635 |
| candidate_constrained_generation | mock | mock-genrec-v1 | 26 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.2454 | 0.1674 | nan | 0.0591 |
| direct_generation | mock | mock-genrec-v1 | 26 | 0.9000 | 0.1000 | 1.0000 | 0.0000 | 0.0457 | 0.0248 | nan | 0.0198 |
| direct_generation | mock | mock-genrec-v1 | 26 | 0.9000 | 0.1000 | 1.0000 | 0.0000 | 0.0457 | 0.0248 | nan | 0.0194 |
| evidence_grounded_llm_explanation | mock | mock-genrec-v1 | 26 | nan | nan | 1.0000 | 0.0000 | nan | nan | 1.0000 | 0.0434 |
| evidence_grounded_llm_explanation | mock | mock-genrec-v1 | 26 | nan | nan | 1.0000 | 0.0000 | nan | nan | 1.0000 | 0.0453 |
| llm_rerank_top_10 | mock | mock-genrec-v1 | 26 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.2410 | 0.1620 | nan | 0.0387 |
| llm_rerank_top_10 | mock | mock-genrec-v1 | 26 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.2410 | 0.1620 | nan | 0.0475 |
| llm_rerank_top_20 | mock | mock-genrec-v1 | 26 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.2390 | 0.1528 | nan | 0.0566 |
| llm_rerank_top_20 | mock | mock-genrec-v1 | 26 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.2390 | 0.1528 | nan | 0.0569 |
| template_explanation | template | template | 26 | nan | nan | 1.0000 | 0.0000 | nan | nan | 1.0000 | 0.0000 |
| template_explanation | template | template | 26 | nan | nan | 1.0000 | 0.0000 | nan | nan | 1.0000 | 0.0000 |

## Direct Generation Risk

Direct generation valid_item_rate=0.9000 and hallucination_rate=0.1000. It is retained only as a baseline for invalid item risk.

Free-form direct generation is not catalog-grounded. It can emit invalid item IDs,
which creates hallucination and serving risk.

## Candidate-Constrained Design

- Selects only from retrieved/ranked catalog candidates.
- Validates every item ID before accepting output.
- Falls back to original ranked candidates if parsing or validation fails.
- Generates explanations from item metadata and candidate evidence only.

## Final Local Decision

`candidate_constrained_generation` is the current local choice among constrained methods: valid_item_rate=1.0000, hallucination_rate=0.0000, parse_success=1.0000, ndcg_at_10=0.2454.

Use LLMs for query understanding, profile summarization, candidate-constrained
reranking/generation, and faithful explanation. Do not use free-form direct
generation as the final recommendation output.

## Not Evaluated Yet

- Real LLM quality.
- Production traffic.
- Human preference evaluation.
- API/dashboard demo.

## Technical Takeaway

“We tested direct generation and candidate-constrained GenRec. Direct generation
had invalid item risk, so the final design grounds LLM outputs in retrieved
catalog candidates and validates every item ID before serving.”
