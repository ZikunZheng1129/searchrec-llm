# Error Taxonomy

## Purpose

Provide a shared vocabulary for future model debugging and error analysis.

## Current Scope

The current project uses synthetic/local debug data and evaluates Stage 3 retrieval baselines plus Stage 4 classic recommendation baselines.

## Retrieval Error Categories

- Lexical mismatch: relevant item uses different wording from the query.
- Category ambiguity: query maps to multiple plausible categories.
- Brand mismatch: retrieved item has the wrong brand.
- Synthetic query artifact: generated query is unnatural or overly templated.
- Cold-start item issue: item has too little metadata or interaction signal.
- Sparse metadata issue: item text lacks enough descriptive terms.

## Recommendation Error Categories

- Popularity bias: model over-recommends globally popular items.
- User history too short: limited train history makes personalization weak.
- Item similarity noise: co-occurrence or content similarity points to weak substitutes.
- Matrix factorization instability: small data creates noisy latent factors.
- Content-only mismatch: text similarity misses behavioral preference.
- Long-tail exposure issue: model fails to surface less common items.

## Future Ranking/LLM Error Categories

- Ranking feature conflict.
- Cross-encoder overfit.
- LLM hallucinated item.
- Unfaithful explanation.
- Constraint parsing error.

## Future Error Analysis CSV Columns

| column | description |
| --- | --- |
| `example_id` | Stable error example ID. |
| `stage` | Pipeline stage being analyzed. |
| `method` | Model or baseline name. |
| `query_or_user_id` | Query ID or user ID. |
| `expected_item_id` | Known relevant item. |
| `predicted_items` | Retrieved or recommended items. |
| `error_category` | Category from this taxonomy. |
| `notes` | Human-readable diagnosis. |
| `proposed_fix` | Candidate improvement or experiment. |
