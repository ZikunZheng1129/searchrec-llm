# LLM Query Understanding

The LLM query-understanding layer adds controlled structured parsing before GenRec. The
goal is structured augmentation: convert natural-language shopping queries into
validated JSON fields that can later improve retrieval, ranking, and generation.

LLMs do not replace retrieval or ranking in this project. They produce structured
signals that are parsed, validated, measured, and kept optional.

## Client Architecture

- `BaseLLMClient`: provider-neutral interface.
- `MockLLMClient`: deterministic local client used by default configs and tests.
- `OpenAIClient`: optional lazy wrapper, disabled unless external calls are
  explicitly allowed.
- `LocalHFClient`: optional lazy local HuggingFace wrapper, disabled unless model
  loading is explicitly allowed.

No API key is required for default workflows.

## JSON Schema

Query understanding returns:

- `intent`
- `category`
- `brand`
- `constraints.price`
- `constraints.use_case`
- `rewritten_query`
- `expanded_queries`
- `confidence`

Pipeline outputs flatten this into columns such as `llm_intent`,
`llm_category`, `llm_price_constraint`, and `llm_use_case`.

## Fallback Behavior

LLM output is parsed with safe JSON utilities. If parsing or schema validation
fails for one query, the pipeline falls back to deterministic rule-based parsing
for that row and records `parse_success=false`.

## Evaluation

Metrics are computed against synthetic labels and rule-based references:

- parse success rate
- schema valid rate
- intent/category/brand/price/use-case match rates
- average expanded query count
- latency
- estimated cost

These metrics validate the local mechanics. They are not production LLM quality
claims.

## Run

```bash
python src/pipelines/evaluate_llm_query_understanding.py --config configs/llm/query_understanding.yaml
python src/pipelines/generate_validation_report.py --config validation/experiments/stage5_debug_validation.yaml
```

Outputs:

- `data/processed/llm_query_understanding.parquet`
- `validation/results/llm_query_understanding_results.csv`

## Caveats

- Synthetic debug data.
- Mock client by default.
- Mock client by default.
- Structured metrics validate mechanics rather than real LLM quality.
- Query examples are synthetic and do not include real query-user logs.

## Future Work

- Real LLM evaluation.
- Prompt ablations.
- Candidate-constrained GenRec.
- Evidence-grounded explanations.
- Hallucination checking for generated recommendations.
