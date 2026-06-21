# User Profile Generation

The user-profile pipeline creates behavior-based profile summaries from observed synthetic interactions and item metadata so downstream components can experiment with personalization.

## Evidence Used

Profiles use only:

- train interactions
- item categories
- item brands
- item prices
- event types
- recent interacted item titles

The profile generator does not infer sensitive traits. It is limited to observed
product behavior.

## Structured Profile Schema

Profiles include:

- `user_id`
- `summary`
- `top_categories`
- `top_brands`
- `price_preference`
- `behavior_signals`
- `recent_interests`
- `profile_text`
- `confidence`

## Mock LLM Generation

The default config uses `MockLLMClient`, which returns deterministic JSON. If an
LLM response fails to parse, the profile generator falls back to deterministic
profile text built from the behavior summary.

## Profile Embeddings

`ProfileEmbedder` builds lightweight bag-of-words vectors from `profile_text`.
No sentence-transformers, external APIs, or neural embedding dependencies are
used.

## Outputs

- `data/processed/user_profiles.parquet`
- `data/processed/user_profile_embeddings.parquet`
- `validation/results/user_profile_results.csv`

## Run

```bash
python src/pipelines/generate_user_profiles.py --config configs/llm/user_profile.yaml
```

## Evaluation

Metrics include:

- profile generation success rate
- schema valid rate
- profile non-empty rate
- embedding coverage
- average profile length
- latency
- estimated cost

## Caveats

- Synthetic debug data.
- Mock client by default.
- Profile quality is not production validated.

## Future Work

- Use profile features in ranking.
- Personalized retrieval.
- Real LLM profile ablations.
- Long-term and short-term preference separation.
