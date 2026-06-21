# Query Generation

The query generation pipeline turns the project into a search-oriented system by creating synthetic search queries from local item metadata.

## Why Synthetic Queries

The debug dataset has users, items, interactions, splits, sequences, and negative samples, but it does not contain search queries. Synthetic query-item pairs let retrieval, ranking, query-understanding, and GenRec components run without downloading a real search log or calling an LLM.

## How Queries Are Generated

The query generator reads `data/processed/item_metadata.parquet` and creates positive query-item pairs from each item. Query templates use item fields such as title, category, brand, price, rating, and category-specific use cases.

Supported query sources include:

- `title`
- `category`
- `brand_category`
- `use_case`
- `price_category`
- `descriptive`

Examples:

- `wireless earbuds`
- `affordable electronics`
- `aster beauty product`
- `running sports and outdoors`
- `budget skincare for dry skin`
- `high rated electronics item`

All generated pairs are positive examples with `relevance_label=1`.

## Query-Item Pair Schema

Saved as `data/processed/query_item_pairs.parquet`.

| Column | Type | Description |
| --- | --- | --- |
| `query_id` | string | Stable unique query-pair identifier. |
| `query_text` | string | Normalized synthetic query text. |
| `target_item_id` | string | Relevant item for the generated query. |
| `category` | string | Parsed or item-derived category. |
| `intent` | string | Rule-based query intent. |
| `source` | string | Template source used to generate the query. |
| `brand` | string | Parsed or item-derived brand. |
| `price_constraint` | string or null | Simple price signal such as `budget` or `premium`. |
| `use_case` | string or null | Simple use case such as `running` or `travel`. |
| `relevance_label` | integer | Positive relevance label, currently always `1`. |
| `split` | string | One of `train`, `val`, or `test`. |

## Rule-Based Intent Classifier

The intent classifier is deterministic and keyword-based. Supported intents are:

- `product_search`
- `brand_search`
- `category_search`
- `price_sensitive_search`
- `use_case_search`
- `unknown`

Price words such as `cheap`, `budget`, `affordable`, and `under` map to `price_sensitive_search`. Use-case words such as `beginner`, `gym`, `running`, `travel`, `dry skin`, `walking`, `outdoor`, `office`, `school`, and `gift` map to `use_case_search`.

## Rule-Based Query Parser

The parser normalizes casing and whitespace, tokenizes the query, and extracts:

- category from known synthetic item categories
- brand from known synthetic brands
- simple price constraints
- simple use-case constraints
- intent through the rule-based classifier

This parser does not call external APIs or LLMs.

## Limitations

The generated queries are synthetic and template-driven, so they do not represent real user search behavior. They are useful for local pipeline validation and future retrieval tests, not for final model claims or business metrics.

Retrieval, ranking, text/metadata representation, and LLM query-understanding workflows use these query-item pairs for evaluation and demo examples.
