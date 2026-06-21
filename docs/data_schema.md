# Data Schema

The data pipeline uses a small synthetic debug dataset. It is designed to exercise local schemas, splits, sequences, and negative sampling without downloading Amazon Reviews 2023 or adding heavy ML dependencies.

## Users

| Column | Type | Description |
| --- | --- | --- |
| `user_id` | string | Stable synthetic user identifier. |
| `user_age_bucket` | string | Coarse synthetic age bucket. |
| `user_region` | string | Coarse synthetic region. |

## Items

Saved as `data/processed/item_metadata.parquet`.

| Column | Type | Description |
| --- | --- | --- |
| `item_id` | string | Stable synthetic item identifier. |
| `title` | string | Fake but realistic product title. |
| `category` | string | Product category. |
| `brand` | string | Synthetic brand name. |
| `price` | float | Synthetic price in dollars. |
| `avg_rating` | float | Synthetic average rating. |
| `rating_count` | integer | Synthetic historical rating count. |
| `description` | string | Fake product description. |

## Interactions

Interactions are generated before splitting and contain implicit-feedback events.

| Column | Type | Description |
| --- | --- | --- |
| `user_id` | string | User identifier. |
| `item_id` | string | Item identifier. |
| `timestamp` | datetime | Event timestamp. |
| `event_type` | string | One of `view`, `click`, `add_to_cart`, or `purchase`. |
| `rating` | float | Synthetic preference signal from 1.0 to 5.0. |
| `event_weight` | float | Numeric event strength. |

Event weights:

| Event | Weight |
| --- | --- |
| `view` | 1.0 |
| `click` | 2.0 |
| `add_to_cart` | 3.0 |
| `purchase` | 4.0 |

## Train, Validation, And Test

Saved as:

- `data/processed/train.parquet`
- `data/processed/val.parquet`
- `data/processed/test.parquet`

These files use the interaction schema plus:

| Column | Type | Description |
| --- | --- | --- |
| `split` | string | One of `train`, `val`, or `test`. |

The current split method is time-aware leave-one-out. For each user, the latest interaction goes to test, the previous interaction goes to validation, and earlier interactions go to train.

## User Sequences

Saved as `data/processed/user_sequences.parquet`.

| Column | Type | Description |
| --- | --- | --- |
| `user_id` | string | User identifier. |
| `item_sequence` | list[string] | Chronological item IDs. |
| `timestamp_sequence` | list[string] | Chronological ISO-format timestamps. |
| `event_type_sequence` | list[string] | Chronological event types. |
| `event_weight_sequence` | list[float] | Chronological event weights. |
| `sequence_length` | integer | Number of items in `item_sequence`. |

## Negative Samples

Saved as `data/processed/negative_samples.parquet`.

| Column | Type | Description |
| --- | --- | --- |
| `user_id` | string | User identifier. |
| `positive_item_id` | string | Item from a positive interaction row. |
| `negative_item_id` | string | Item the user has not interacted with. |
| `split` | string | Split associated with the positive interaction. |

## Notes

The synthetic debug data is deterministic for a fixed seed and is intentionally small enough for quick local tests. Real public-data processing is future work.
