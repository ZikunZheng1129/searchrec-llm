# Interview Q&A

## 1. How does hybrid retrieval work in your system?

It combines BM25 sparse scores and TF-IDF dense cosine scores with configurable weights and per-query score normalization. The goal is to capture both lexical matches and broader token-overlap similarity.

## 2. Why did you implement BM25 before dense retrieval?

BM25 is a strong, interpretable search baseline. It gives a useful lexical floor and helps identify whether later dense or hybrid methods actually improve retrieval.

## 3. How do you evaluate retrieval?

I use Recall@K, MRR@K, query coverage, average latency, and p95 latency over synthetic query-item pairs.

## 4. How do you evaluate recommendation baselines?

I use HitRate@K, Recall@K, NDCG@K, MRR@K, coverage, and latency over held-out user interactions.

## 5. Why include both recommendation and retrieval?

Search and recommendation answer different product questions. Retrieval is query-driven candidate recall, while recommendation is user-driven candidate generation. The final system benefits from both.

## 6. Why add sequential models?

Sequential models capture order-sensitive user behavior that classic recommenders may miss. GRU4Rec and SASRec provide recurrent and attention-based baselines.

## 7. How do you evaluate ranking?

Ranking is evaluated with NDCG, MRR, Precision, Recall, AUC, candidate coverage, and latency on query-item candidate sets.

## 8. Why did the LightGBM-style ranker use a fallback?

The project is local-first. The wrapper preserves a production-shaped interface while keeping tests runnable without requiring every optional package.

## 9. What is candidate-constrained GenRec?

It gives the generator a known candidate set and only accepts output item IDs from that set or catalog. This grounds generative recommendations in available items.

## 10. Why is candidate-constrained GenRec safer?

It reduces invalid item recommendations. In the current synthetic debug results, constrained generation has valid_item_rate 1.000000 and hallucination_rate 0.000000, while direct generation has valid_item_rate 0.900000 and hallucination_rate 0.100000.

## 11. How do you prevent hallucinated recommendations?

The GenRec layer parses outputs, checks schema validity, validates item IDs against candidates/catalog entries, measures hallucination rate, and falls back to grounded candidates when needed.

## 12. How does LLM query understanding work?

The pipeline prompts a mock-first client to return structured JSON for intent, category, brand, price constraints, use case, and query expansions. The output is schema-validated.

## 13. Why use mock LLM outputs?

Mock clients make the demo deterministic, free, safe, and runnable without API keys. Optional provider wrappers exist for future evaluation.

## 14. How do user profiles work?

User profiles summarize behavior from local histories and produce profile embeddings that can later feed personalization and ranking features.

## 15. What is multimodal in this project?

The current implemented signals are text and metadata, with an optional image interface. The system is ready for real image embeddings, but the debug dataset does not include them.

## 16. How would you handle cold start?

Use item text, metadata, category, brand, and future image embeddings to retrieve and rank new items without needing many interactions.

## 17. How would you scale retrieval to millions of items?

Use offline embedding generation, FAISS or another ANN system, sharded indexes, candidate caching, metadata filters, and latency budgets.

## 18. How would this change with real query logs?

I would add time-based splits, session context, impression/click/purchase labels, query reformulation signals, and online-safe evaluation slices.

## 19. How would you use Amazon Reviews 2023 data?

I would build a public-data ingestion pipeline, map reviews to interactions, derive item metadata, generate or use query signals where available, and rerun every validation component.

## 20. What would you do differently in production?

I would add privacy controls, data quality checks, feature stores, model monitoring, ANN serving, online experiments, human evaluation for explanations, cost controls for LLMs, and robust alerting.

## 21. Why not compare all models with one global score?

Retrieval, recommendation, ranking, text/metadata retrieval, and GenRec solve different tasks. The leaderboard selects the best row per component but does not claim a global winner.

## 22. What is the most important engineering choice?

Keeping LLM outputs grounded in retrieved/ranked catalog candidates. It preserves the benefits of generation while respecting catalog validity.
