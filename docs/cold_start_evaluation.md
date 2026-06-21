# Cold-Start Evaluation

Cold-start items have little or no behavior data, so content representations are
important for candidate recall and ranking.

## Current Definition

The local debug dataset is synthetic, so the text/metadata representation workflow simulates cold-start and
long-tail slices using low train interaction counts.

- Cold-start items: bottom quantile by train interaction count.
- Long-tail items: bottom quantile by train interaction count.

When a slice would be empty, the lowest-count item set is used when possible.

## Metrics

- `cold_start_recall_at_k`: Recall@K averaged over queries whose target item is
  in the simulated cold-start slice.
- `long_tail_coverage_at_k`: unique long-tail items retrieved in top-K divided by
  the number of long-tail items.
- `catalog_coverage_at_k`: unique retrieved items in top-K divided by catalog
  size.
- `category_diversity_at_k`: average unique categories in top-K normalized by the
  smaller of K and available category count.

## Limitations

These metrics are useful for local validation mechanics, not production
cold-start performance. Real evaluation needs true new items, production query
logs, richer metadata, and real item images or precomputed image embeddings.
