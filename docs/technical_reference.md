# Technical Reference

This reference documents the current inspected SearchRec-LLM repository. It is meant for maintainers and technical readers; the README stays focused on the public project overview.

All paths are repository-relative. Current artifacts are built from synthetic debug data and deterministic mock LLM clients unless noted otherwise.

## Repository Overview

SearchRec-LLM is a local-first search, recommendation, ranking, structured-LLM, text-and-metadata representation, GenRec, API, and dashboard project.

Top-level layout:

| Path | Responsibility |
| --- | --- |
| `configs/` | YAML configs for data, retrieval, recommendation, sequence models, ranking, LLM, GenRec, text/metadata representation, API, and dashboard. |
| `src/data/` | Synthetic dataset generation, schema validation, splitting, negative sampling, and query generation. |
| `src/query_understanding/` | Deterministic rule-based intent classification, parsing, and bag-of-words query encoding. |
| `src/retrieval/` | BM25, TF-IDF dense, FAISS-style vector wrapper, and hybrid retrieval. |
| `src/recommendation/` | Popularity, itemCF, matrix factorization, and user-history embedding recommenders. |
| `src/sequence_models/` | GRU4Rec, SASRec, sequence datasets, sampled BCE loss, training, and evaluation. |
| `src/ranking/` | Ranking features, candidate datasets, rankers, and ranker artifact IO. |
| `src/llm/` | Mock-first LLM clients, structured query understanding, user profiles, GenRec, reranking, parsing, explanations, and hallucination checks. |
| `src/multimodal/` | Text encoders, metadata encoders, optional image-feature interfaces, fusion, retrieval, and cold-start slices. |
| `src/evaluation/` | Component metrics and validation report generation. |
| `src/pipelines/` | Command-line pipeline entry points. |
| `app/api/` | Local FastAPI service over generated artifacts. |
| `app/dashboard/` | Streamlit demo over generated artifacts. |
| `validation/` | Experiment templates, result CSVs, reports, ablations, and error analysis. |
| `docs/` | Public technical documentation, results summaries, reproducibility notes, and demo guides. |
| `tests/` | Unit and integration tests. |

## Package And Module Map

### `src/utils/config.py`

**Responsibility:** YAML config loading/saving and repository-relative path resolution.

**Public API**
- `load_yaml_config(path: str | Path) -> dict[str, Any]`
- `save_yaml_config(config: dict[str, Any], path: str | Path) -> None`
- `resolve_project_path(*parts: str) -> Path`

**Inputs:** YAML file paths and repository-relative path parts.

**Outputs:** Python dictionaries, written YAML files, and resolved `Path` objects.

**Used by:** Most pipeline entry points and app config loading.

### `src/utils/io.py`

**Responsibility:** Small filesystem and serialization helpers.

**Public API**
- `ensure_dir(path: str | Path) -> Path`
- `read_json(path: str | Path) -> dict[str, Any]`
- `write_json(obj: dict[str, Any], path: str | Path) -> None`
- `read_parquet(path: str | Path) -> pd.DataFrame`
- `write_parquet(df: pd.DataFrame, path: str | Path) -> None`

**Used by:** Data and artifact pipelines.

### `src/utils/logging.py`

**Public API**
- `get_logger(name: str, level: str = "INFO") -> logging.Logger`

### `src/utils/seed.py`

**Public API**
- `set_seed(seed: int = 42) -> None`

### `src/data/dataset.py`

**Responsibility:** DataFrame schema validation and split labeling.

**Public API**
- `validate_columns(df, required_columns, name) -> None`
- `validate_interactions_schema(interactions) -> None`
- `validate_items_schema(items) -> None`
- `validate_users_schema(users) -> None`
- `attach_split_column(train, val, test) -> pd.DataFrame`

**Inputs:** User, item, and interaction DataFrames.

**Outputs:** Validation errors or combined split-labeled interactions.

**Used by:** Dataset build and tests.

### `src/data/preprocess.py`

**Responsibility:** Deterministic synthetic data generation and cleaning.

**Public API**
- `generate_synthetic_items(num_items, categories, seed=42) -> pd.DataFrame`
- `generate_synthetic_users(num_users, seed=42) -> pd.DataFrame`
- `generate_synthetic_interactions(users, items, num_interactions, start_timestamp, end_timestamp, seed=42) -> pd.DataFrame`
- `clean_items(items) -> pd.DataFrame`
- `clean_users(users) -> pd.DataFrame`
- `clean_interactions(interactions) -> pd.DataFrame`
- `build_debug_dataset(config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`

**Inputs:** Synthetic generation config.

**Outputs:** Users, items, and interactions.

**Used by:** `src/pipelines/build_dataset.py`.

### `src/data/split.py`

**Responsibility:** Time-aware leave-one-out splitting and user sequence construction.

**Public API**
- `leave_one_out_split(interactions, val_last_n=1, test_last_n=1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`
- `build_user_sequences(interactions) -> pd.DataFrame`

**Outputs:** Train/validation/test interactions and chronological user sequences.

### `src/data/negative_sampling.py`

**Public API**
- `sample_negative_items(interactions, all_item_ids, num_negatives_per_positive=5, seed=42) -> pd.DataFrame`

**Outputs:** Negative samples with `user_id`, `positive_item_id`, `negative_item_id`, and `split`.

### `src/data/query_generation.py`

**Responsibility:** Template-based synthetic search query generation.

**Public API**
- `generate_queries_for_item(item, queries_per_item=3, seed=42) -> list[dict[str, Any]]`
- `generate_query_item_pairs(items, config, seed=42) -> pd.DataFrame`
- `add_query_splits(query_item_pairs, seed=42) -> pd.DataFrame`
- `validate_query_item_pairs(query_item_pairs) -> None`

**Inputs:** Item metadata.

**Outputs:** Positive query-item pairs with query metadata and split labels.

**Used by:** Retrieval, ranking, text/metadata retrieval, LLM query understanding, and GenRec validation.

### `src/query_understanding/intent_classifier.py`

**Public API**
- `classify_intent(query_text: str) -> str`

**Responsibility:** Deterministic keyword intent labels.

### `src/query_understanding/query_parser.py`

**Public API**
- `normalize_query(query_text: str) -> str`
- `extract_price_constraint(query_text: str) -> str | None`
- `extract_use_case(query_text: str) -> str | None`
- `extract_category(query_text, known_categories) -> str | None`
- `extract_brand(query_text, known_brands) -> str | None`
- `parse_query(query_text, known_categories=None, known_brands=None) -> dict[str, Any]`

**Outputs:** Rule-based intent, category, brand, price, and use-case fields.

### `src/query_understanding/query_encoder.py`

**Public API**
- `BagOfWordsQueryEncoder.fit(texts)`
- `BagOfWordsQueryEncoder.transform(texts)`
- `BagOfWordsQueryEncoder.fit_transform(texts)`
- `BagOfWordsQueryEncoder.get_feature_names()`

### `src/retrieval/text_utils.py`

**Public API**
- `normalize_text(text: str) -> str`
- `tokenize(text: str) -> list[str]`
- `build_item_text(row: pd.Series, fields: Sequence[str] | None = None) -> str`

**Responsibility:** Shared item text normalization over title/category/brand/description.

### `src/retrieval/bm25_retriever.py`

**Responsibility:** Local lexical candidate retrieval.

**Public API**
- `BM25Retriever.fit(items: pd.DataFrame) -> BM25Retriever`
- `BM25Retriever.search(query: str, top_k: int = 50) -> list[dict[str, Any]]`
- `BM25Retriever.batch_search(queries: list[str], top_k: int = 50) -> dict[str, list[dict[str, Any]]]`
- `BM25Retriever.save(path: str | Path) -> None`
- `BM25Retriever.load(path: str | Path) -> BM25Retriever`

**Inputs:** Item metadata.

**Outputs:** Ranked item IDs with scores/ranks.

**Used by:** Retrieval evaluation and ranking candidate generation.

### `src/retrieval/dense_retriever.py`

**Responsibility:** Local TF-IDF vector retrieval with cosine similarity.

**Public API**
- `TfidfDenseRetriever.fit(items) -> TfidfDenseRetriever`
- `TfidfDenseRetriever.encode_queries(queries) -> np.ndarray`
- `TfidfDenseRetriever.search(query, top_k=50) -> list[dict[str, Any]]`
- `TfidfDenseRetriever.batch_search(queries, top_k=50) -> dict[str, list[dict[str, Any]]]`
- `TfidfDenseRetriever.save(path) -> None`
- `TfidfDenseRetriever.load(path) -> TfidfDenseRetriever`

**Observed backend:** `numpy`.

### `src/retrieval/faiss_retriever.py`

**Responsibility:** FAISS-shaped vector retrieval interface over TF-IDF vectors.

**Public API**
- `FaissRetriever.fit(items) -> FaissRetriever`
- `FaissRetriever.search(query, top_k=50) -> list[dict[str, Any]]`
- `FaissRetriever.batch_search(queries, top_k=50) -> dict[str, list[dict[str, Any]]]`
- `FaissRetriever.save(path) -> None`
- `FaissRetriever.load(path) -> FaissRetriever`

**Fallback behavior:** In the inspected environment, `faiss` is not installed and the reported backend is `numpy`.

### `src/retrieval/hybrid_retriever.py`

**Responsibility:** Sparse+dense score fusion.

**Public API**
- `HybridRetriever.fit(items) -> HybridRetriever`
- `HybridRetriever.search(query, top_k=50) -> list[dict[str, Any]]`
- `HybridRetriever.batch_search(queries, top_k=50) -> dict[str, list[dict[str, Any]]]`
- `HybridRetriever.save(path) -> None`
- `HybridRetriever.load(path) -> HybridRetriever`

**Observed backend:** `bm25+dense_numpy`.

### `src/recommendation/base.py`

**Public API**
- `BaseRecommender.fit(train_interactions, items=None) -> BaseRecommender`
- `BaseRecommender.recommend(user_id, top_k=10, exclude_seen=True) -> list[dict[str, Any]]`
- `BaseRecommender.batch_recommend(user_ids, top_k=10, exclude_seen=True) -> dict[str, list[dict[str, Any]]]`

### `src/recommendation/popularity.py`

**Public API**
- `PopularityRecommender.fit(train_interactions, items=None) -> PopularityRecommender`
- `PopularityRecommender.recommend(user_id, top_k=10, exclude_seen=True) -> list[dict[str, Any]]`

**Responsibility:** Global weighted-interaction baseline.

### `src/recommendation/itemcf.py`

**Public API**
- `ItemCFRecommender.fit(train_interactions, items=None) -> ItemCFRecommender`
- `ItemCFRecommender.recommend(user_id, top_k=10, exclude_seen=True) -> list[dict[str, Any]]`

**Responsibility:** Item-item cosine similarity over implicit interactions.

### `src/recommendation/matrix_factorization.py`

**Public API**
- `MatrixFactorizationRecommender.fit(train_interactions, items=None) -> MatrixFactorizationRecommender`
- `MatrixFactorizationRecommender.recommend(user_id, top_k=10, exclude_seen=True) -> list[dict[str, Any]]`

**Responsibility:** Small NumPy matrix factorization baseline.

### `src/recommendation/user_history_embedding.py`

**Public API**
- `UserHistoryEmbeddingRecommender.fit(train_interactions, items=None) -> UserHistoryEmbeddingRecommender`
- `UserHistoryEmbeddingRecommender.recommend(user_id, top_k=10, exclude_seen=True) -> list[dict[str, Any]]`

**Responsibility:** User profile vector from weighted item text vectors.

### `src/features/user_features.py`

**Public API**
- `build_user_item_history(interactions) -> dict[str, set[str]]`
- `build_user_event_weights(interactions) -> dict[str, dict[str, float]]`
- `get_seen_items(user_id, user_history) -> set[str]`
- `build_user_profile_summary(user_id, interactions, items) -> dict[str, Any]`

### `src/sequence_models/dataset.py`

**Public API**
- `SequenceMappings.num_items() -> int`
- `build_item_id_mapping(items, interactions) -> SequenceMappings`
- `encode_interactions(interactions, mappings) -> pd.DataFrame`
- `build_train_examples(train_interactions, mappings, max_seq_len, min_seq_len=2) -> list[dict[str, Any]]`
- `build_eval_examples(train_interactions, eval_interactions, mappings, max_seq_len) -> list[dict[str, Any]]`
- `SequenceTrainDataset`
- `SequenceEvalDataset`
- `sequence_collate_fn(batch) -> dict[str, Any]`

**Responsibility:** Map item IDs to integer IDs with padding index 0 and create prefix-to-next-item examples.

### `src/sequence_models/gru4rec.py` and `src/sequence_models/sasrec.py`

**Public API**
- `GRU4Rec.forward(input_ids)`
- `GRU4Rec.score_items(user_repr, item_ids=None)`
- `GRU4Rec.recommend(input_ids, top_k=10, exclude_ids=None)`
- `SASRec.forward(input_ids)`
- `SASRec.score_items(user_repr, item_ids=None)`
- `SASRec.recommend(input_ids, top_k=10, exclude_ids=None)`

**Responsibility:** Local next-item sequence models.

### `src/sequence_models/losses.py`

**Public API**
- `sample_negative_item_ids(positive_ids, num_items, num_negatives, generator=None) -> torch.Tensor`
- `sampled_bce_loss(positive_scores, negative_scores) -> torch.Tensor`

### `src/sequence_models/trainer.py`

**Public API**
- `resolve_path(path) -> Path`
- `choose_device(device_config) -> torch.device`
- `create_sequence_model(method, num_items, config) -> torch.nn.Module`
- `train_one_epoch(...) -> float`
- `evaluate_loss(...) -> float`
- `train_sequence_model(config) -> dict[str, Any]`

### `src/sequence_models/evaluate.py`

**Public API**
- `load_sequence_checkpoint(path, device="cpu") -> tuple[torch.nn.Module, SequenceMappings, dict[str, Any]]`
- `evaluate_sequence_model(model, train_interactions, eval_interactions, items, mappings, config) -> pd.DataFrame`
- `summarize_sequence_metrics(per_user_results) -> pd.DataFrame`

### `src/ranking/features.py`

**Responsibility:** Build query-candidate ranking features.

**Public API**
- `build_candidate_item_text(items) -> pd.DataFrame`
- `compute_item_popularity_features(train_interactions) -> pd.DataFrame`
- `compute_conversion_proxy(train_interactions) -> pd.DataFrame`
- `compute_price_bucket(price) -> str`
- `compute_authority_score(avg_rating, rating_count) -> float`
- `compute_business_proxy_features(train_interactions, items) -> pd.DataFrame`
- `compute_text_match_features(query_text, item_row) -> dict[str, float]`
- `build_ranking_features(query_item_pairs, items, train_interactions, candidate_pool_size, random_negatives_per_query, seed, config) -> pd.DataFrame`

**Main features:** BM25/dense/hybrid scores and ranks, category/brand match, token overlap, popularity/event weights, purchase/add-to-cart counts, rating/price fields, authority proxy, conversion proxy, cold-start score, and diversity category.

### `src/ranking/dataset.py`

**Public API**
- `validate_ranking_candidates(df) -> None`
- `get_numeric_feature_columns(df) -> list[str]`
- `split_ranking_candidates(df, split) -> pd.DataFrame`
- `apply_feature_standardizer(df, standardizer) -> pd.DataFrame`
- `standardize_features(train_df, other_df) -> tuple[pd.DataFrame, dict[str, Any]]`
- `RankingFeatureDataset`

### `src/ranking/feature_ranker.py`

**Public API**
- `NumpyLinearRanker.fit(train_df, feature_columns, label_column="label") -> NumpyLinearRanker`
- `NumpyLinearRanker.predict(df) -> np.ndarray`
- `NumpyLinearRanker.save(path) -> None`
- `NumpyLinearRanker.load(path) -> NumpyLinearRanker`

### `src/ranking/lightgbm_ranker.py`

**Public API**
- `LightGBMRanker.fit(train_df, feature_columns, group_column="query_id", label_column="label") -> LightGBMRanker`
- `LightGBMRanker.predict(df) -> np.ndarray`
- `LightGBMRanker.save(path) -> None`
- `LightGBMRanker.load(path) -> LightGBMRanker`

**Fallback behavior:** In the inspected environment, `lightgbm` is not installed and the reported backend is `numpy_linear`.

### `src/ranking/mlp_ranker.py`

**Public API**
- `MLPRanker.forward(features) -> torch.Tensor`
- `MLPRankerWrapper.fit(train_df, feature_columns, label_column="label") -> MLPRankerWrapper`
- `MLPRankerWrapper.predict(df) -> np.ndarray`
- `MLPRankerWrapper.save(path) -> None`
- `MLPRankerWrapper.load(path, device="cpu") -> MLPRankerWrapper`

### `src/ranking/cross_encoder_ranker.py`

**Public API**
- `PairTextVectorizer.fit(df, items) -> PairTextVectorizer`
- `PairTextVectorizer.encode_pair(query_text, item_text) -> list[int]`
- `PairTextVectorizer.to_dict() -> dict[str, Any]`
- `PairTextVectorizer.from_dict(state) -> PairTextVectorizer`
- `CrossEncoderDataset`
- `TinyCrossEncoderRanker.forward(input_ids) -> torch.Tensor`
- `CrossEncoderRankerWrapper.fit(train_df, items, label_column="label") -> CrossEncoderRankerWrapper`
- `CrossEncoderRankerWrapper.predict(df, items) -> np.ndarray`
- `CrossEncoderRankerWrapper.save(path) -> None`
- `CrossEncoderRankerWrapper.load(path, device="cpu") -> CrossEncoderRankerWrapper`

### `src/ranking/mixed_ranker.py`

**Public API**
- `MixedRanker.fit(train_df=None) -> MixedRanker`
- `MixedRanker.predict(df) -> np.ndarray`
- `MixedRanker.rerank(df, top_k) -> pd.DataFrame`
- `MixedRanker.save(path) -> None`
- `MixedRanker.load(path) -> MixedRanker`

### `src/ranking/io.py`

**Public API**
- `save_ranker_artifact(obj, path) -> None`
- `load_ranker_artifact(path) -> Any`
- `update_ranking_results(results_row, results_path) -> None`

### `src/llm/clients/*`

**Responsibility:** Provider-neutral LLM interface and optional providers.

**Public API**
- `LLMResponse`
- `BaseLLMClient.generate(prompt, system_prompt=None, temperature=0.0, max_tokens=512, response_format="json") -> LLMResponse`
- `MockLLMClient.generate(...) -> LLMResponse`
- `OpenAIClient.generate(...) -> LLMResponse`
- `LocalHFClient.generate(...) -> LLMResponse`

**Observed runtime:** `openai` and `transformers` are not installed. Default configs use `MockLLMClient` with `allow_external_api_calls: false`.

### `src/llm/utils.py`

**Public API**
- `extract_json_object(text) -> str`
- `safe_json_loads(text) -> tuple[dict[str, Any], bool, str | None]`
- `validate_required_keys(obj, required_keys, name) -> None`
- `estimate_mock_cost(prompt_tokens, completion_tokens, provider, model) -> float`
- `estimate_token_count(text) -> int`

### `src/llm/query_understanding/*`

**Public API**
- `QueryUnderstandingResult`
- `validate_query_understanding_result(obj) -> dict[str, Any]`
- `LLMIntentExtractor.parse_query(query_text) -> dict[str, Any]`
- `LLMIntentExtractor.batch_parse_queries(query_texts) -> list[dict[str, Any]]`
- `normalize_expanded_queries(values) -> list[str]`
- `fallback_query_rewrite(query_text, parsed_query) -> str`
- `fallback_query_expansion(query_text, parsed_query, max_queries=3) -> list[str]`
- `build_llm_client_from_config(config) -> BaseLLMClient`
- `run_query_understanding(query_item_pairs, items, config) -> pd.DataFrame`

**Outputs:** `data/processed/llm_query_understanding.parquet`.

### `src/llm/user_modeling/*`

**Public API**
- `build_user_behavior_summary(user_id, train_interactions, items, max_recent_items=5) -> dict[str, Any]`
- `build_all_user_behavior_summaries(...) -> list[dict[str, Any]]`
- `LLMUserProfileGenerator.generate_profile(behavior_summary) -> dict[str, Any]`
- `LLMUserProfileGenerator.batch_generate_profiles(behavior_summaries) -> pd.DataFrame`
- `ProfileEmbedder.transform(profile_texts) -> np.ndarray`
- `ProfileEmbedder.fit_transform(profile_texts) -> np.ndarray`
- `ProfileEmbedder.get_feature_names() -> list[str]`
- `ProfileEmbedder.save(path) -> None`
- `ProfileEmbedder.load(path) -> ProfileEmbedder`
- `run_user_profile_generation(train_interactions, items, config) -> tuple[pd.DataFrame, pd.DataFrame]`

**Outputs:** `data/processed/user_profiles.parquet` and `data/processed/user_profile_embeddings.parquet`.

### `src/llm/genrec/*`

**Public API**
- `load_candidate_source(config) -> pd.DataFrame`
- `select_top_candidates_for_query(ranking_candidates, query_id, top_n, score_priority=None) -> pd.DataFrame`
- `format_candidate_for_llm(row, item_row=None) -> dict[str, Any]`
- `format_candidates_for_llm(candidates, items, max_candidates=20) -> list[dict[str, Any]]`
- `build_candidate_lookup(candidates) -> dict[str, dict]`
- `DirectGenerator.generate(query_text, parsed_query=None, user_profile=None, top_k=10) -> dict[str, Any]`
- `DirectGenerator.batch_generate(contexts, top_k=10) -> pd.DataFrame`
- `CandidateConstrainedGenerator.generate(query_text, candidates, parsed_query=None, user_profile=None, top_k=10) -> dict[str, Any]`
- `CandidateConstrainedGenerator.batch_generate(contexts, top_k=10) -> pd.DataFrame`
- `LLMReranker.rerank(query_text, candidates, parsed_query=None, user_profile=None, top_k=10) -> dict[str, Any]`
- `LLMReranker.batch_rerank(contexts, top_k=10) -> pd.DataFrame`
- `parse_recommendation_output(text) -> dict[str, Any]`
- `parse_rerank_output(text) -> dict[str, Any]`
- `parse_explanation_output(text) -> dict[str, Any]`
- `normalize_recommendation_items(obj, candidate_item_ids=None, top_k=10) -> dict[str, Any]`
- `validate_candidate_constrained_output(obj, candidate_item_ids) -> dict[str, Any]`
- `fallback_recommendations_from_candidates(candidates, top_k=10) -> dict[str, Any]`
- `run_genrec_pipeline(config, config_path=None) -> dict[str, Any]`

**Outputs:** GenRec, rerank, explanation, and result artifacts under `data/processed/` and `validation/results/`.

### `src/llm/explanations/*`

**Public API**
- `build_evidence_text(evidence) -> str`
- `select_item_evidence(item_id, item_metadata, ranking_candidates=None) -> dict[str, Any]`
- `select_evidence_for_recommendations(recommended_item_ids, items, ranking_candidates_for_query=None, max_items=5) -> list[dict[str, Any]]`
- `TemplateExplanationGenerator.generate(...) -> dict[str, Any]`
- `LLMExplanationGenerator.generate(...) -> dict[str, Any]`
- `check_valid_item_ids(output_item_ids, valid_item_ids) -> dict[str, Any]`
- `check_candidate_constrained_item_ids(output_item_ids, candidate_item_ids) -> dict[str, Any]`
- `check_explanation_faithfulness(explanations, evidence_by_item_id) -> dict[str, Any]`
- `compute_hallucination_flags(row) -> dict[str, Any]`

### `src/multimodal/text_encoder.py`

**Public API**
- `TextItemEncoder.fit(items) -> TextItemEncoder`
- `TextItemEncoder.transform(items) -> np.ndarray`
- `TextItemEncoder.fit_transform(items) -> np.ndarray`
- `TextItemEncoder.encode_queries(query_texts) -> np.ndarray`
- `TextItemEncoder.get_feature_names() -> list[str]`
- `TextItemEncoder.save(path) -> None`
- `TextItemEncoder.load(path) -> TextItemEncoder`

**Observed output dimension:** 237.

### `src/multimodal/metadata_encoder.py`

**Public API**
- `MetadataItemEncoder.fit(items) -> MetadataItemEncoder`
- `MetadataItemEncoder.transform(items) -> np.ndarray`
- `MetadataItemEncoder.fit_transform(items) -> np.ndarray`
- `MetadataItemEncoder.encode_queries(query_texts) -> np.ndarray`
- `MetadataItemEncoder.get_feature_names() -> list[str]`
- `MetadataItemEncoder.save(path) -> None`
- `MetadataItemEncoder.load(path) -> MetadataItemEncoder`

**Observed output dimension:** 17.

### `src/multimodal/image_encoder.py`

**Public API**
- `ImageItemEncoder.fit(items) -> ImageItemEncoder`
- `ImageItemEncoder.transform(items) -> np.ndarray | None`
- `ImageItemEncoder.fit_transform(items) -> np.ndarray | None`
- `ImageItemEncoder.image_available_rate(items) -> float`
- `PrecomputedImageFeatureEncoder`
- `TestStubImageEncoder`

**Observed status:** The current data has no real image embeddings and image availability is 0.0.

### `src/multimodal/fusion_model.py`

**Public API**
- `MultimodalFusionEncoder.fit(items) -> MultimodalFusionEncoder`
- `MultimodalFusionEncoder.transform(items) -> np.ndarray`
- `MultimodalFusionEncoder.fit_transform(items) -> np.ndarray`
- `MultimodalFusionEncoder.encode_queries(query_texts) -> np.ndarray`
- `MultimodalFusionEncoder.save(path) -> None`
- `MultimodalFusionEncoder.load(path) -> MultimodalFusionEncoder`
- `build_image_encoder_from_config(config) -> Any | None`
- `build_encoder_from_config(config) -> Any`

**Observed fused dimension:** 254 for text + metadata.

### `src/multimodal/multimodal_retriever.py`

**Public API**
- `MultimodalRetriever.fit(items, embeddings, query_encoder) -> MultimodalRetriever`
- `MultimodalRetriever.search(query, top_k=50) -> list[dict[str, Any]]`
- `MultimodalRetriever.batch_search(queries, top_k=50) -> dict[str, list[dict[str, Any]]]`
- `MultimodalRetriever.save(path) -> None`
- `MultimodalRetriever.load(path) -> MultimodalRetriever`

### `src/multimodal/cold_start_model.py`

**Public API**
- `compute_item_interaction_counts(train_interactions) -> pd.DataFrame`
- `identify_cold_start_items(items, train_interactions, quantile=0.25) -> set[str]`
- `identify_long_tail_items(items, train_interactions, quantile=0.25) -> set[str]`
- `add_cold_start_flags(items, train_interactions, config) -> pd.DataFrame`

### `src/multimodal/ranking_features.py`

**Public API**
- `add_multimodal_scores_to_ranking_candidates(ranking_candidates, query_item_pairs, items, retriever, score_column="multimodal_score", rank_column="multimodal_rank") -> pd.DataFrame`

### `src/evaluation/*`

**Responsibility:** Component-specific metrics and final validation report generation.

**Major public functions**
- Retrieval: `recall_at_k`, `mrr_at_k`, `evaluate_retriever`, `summarize_retrieval_metrics`
- Recommendation: `hit_rate_at_k`, `recall_at_k`, `ndcg_at_k`, `mrr_at_k`, `coverage_at_k`, `evaluate_recommender`, `summarize_recommendation_metrics`
- Sequence: `summarize_sequence_metrics`
- Ranking: `dcg_at_k`, `ndcg_at_k`, `mrr_at_k`, `precision_at_k`, `recall_at_k`, `auc_score`, `evaluate_ranking_predictions`, `summarize_ranking_metrics`
- LLM: `evaluate_query_understanding_outputs`, `evaluate_user_profile_outputs`, `summarize_llm_metrics`
- Text/metadata representation: `evaluate_multimodal_retriever`, `summarize_multimodal_metrics`
- GenRec: `valid_item_rate`, `hallucination_rate`, `evaluate_genrec_outputs`, `evaluate_explanations`, `summarize_genrec_metrics`
- Reporting: `build_final_leaderboard`, `select_best_models`, `write_leaderboard`, `generate_model_selection_report`, `generate_latency_quality_report`, `generate_validation_summary`, `generate_validation_artifacts`

**Important principle:** Metrics are component-specific and should not be compared globally across retrieval, recommendation, ranking, and GenRec tasks.

## Pipeline CLI Reference

Each pipeline accepts `--config`.

| Command | Purpose | Typical output |
| --- | --- | --- |
| `python3 src/pipelines/build_dataset.py --config configs/data/debug_sample.yaml` | Build synthetic users/items/interactions, splits, sequences, negatives. | `data/processed/*.parquet` |
| `python3 src/pipelines/generate_queries.py --config configs/data/query_generation_debug.yaml` | Build synthetic query-item pairs. | `data/processed/query_item_pairs.parquet` |
| `python3 src/pipelines/build_index.py --config configs/retrieval/bm25_debug.yaml` | Build a retrieval index. | `data/indexes/bm25_debug.pkl` |
| `python3 src/pipelines/evaluate_retrieval.py --config configs/retrieval/bm25_debug.yaml` | Evaluate retrieval. | `validation/results/retrieval_results.csv` |
| `python3 src/pipelines/evaluate_recommenders.py --config configs/recommendation/itemcf_debug.yaml` | Evaluate recommender baselines. | `validation/results/recommender_baselines.csv` |
| `python3 src/pipelines/train_sequence_model.py --config configs/sequence/sasrec_debug.yaml` | Train a sequence model. | `artifacts/models/sequence/sasrec_debug.pt` |
| `python3 src/pipelines/evaluate_sequence_model.py --config configs/sequence/sasrec_debug.yaml` | Evaluate a sequence model. | `validation/results/sequence_results.csv` |
| `python3 src/pipelines/build_ranking_dataset.py --config configs/ranking/ranking_dataset_debug.yaml` | Build query-candidate ranking rows. | `data/processed/ranking_candidates.parquet` |
| `python3 src/pipelines/train_ranker.py --config configs/ranking/lightgbm_ranker_debug.yaml` | Train a ranker. | `artifacts/models/ranking/lightgbm_ranker_debug.pkl` |
| `python3 src/pipelines/evaluate_ranker.py --config configs/ranking/lightgbm_ranker_debug.yaml` | Evaluate a ranker. | `validation/results/ranking_results.csv` |
| `python3 src/pipelines/evaluate_llm_query_understanding.py --config configs/llm/query_understanding.yaml` | Run mock structured query understanding. | `data/processed/llm_query_understanding.parquet` |
| `python3 src/pipelines/generate_user_profiles.py --config configs/llm/user_profile.yaml` | Generate mock behavior-grounded profiles and profile embeddings. | `data/processed/user_profiles.parquet` |
| `python3 src/pipelines/build_multimodal_embeddings.py --config configs/multimodal/text_metadata_fusion_debug.yaml` | Build text/metadata item embeddings. | `data/embeddings/item_multimodal_embeddings.parquet` |
| `python3 src/pipelines/evaluate_multimodal.py --config configs/multimodal/text_metadata_fusion_debug.yaml` | Evaluate text/metadata retrieval. | `validation/results/multimodal_results.csv` |
| `python3 src/pipelines/augment_ranking_with_multimodal.py --config configs/multimodal/multimodal_fusion_debug.yaml` | Add representation scores to ranking candidates. | `data/processed/ranking_candidates_multimodal.parquet` |
| `python3 src/pipelines/evaluate_genrec.py --config configs/genrec/genrec_debug.yaml` | Run mock GenRec, rerank, and explanation experiments. | `validation/results/genrec_results.csv` |
| `python3 src/pipelines/generate_validation_report.py --config validation/experiments/stage5_debug_validation.yaml` | Build final leaderboard and reports from existing result CSVs. | `validation/results/final_leaderboard.csv`, `validation/reports/*.md` |

Script wrappers:

- `bash scripts/run_validation.sh`: bounded local validation workflow; optional heavier groups are gated by `RUN_RANKING=1`, `RUN_LLM=1`, `RUN_MULTIMODAL=1`, and `RUN_GENREC=1`.
- `bash scripts/launch_api.sh`: starts FastAPI on `http://127.0.0.1:8000`.
- `bash scripts/launch_dashboard.sh`: starts Streamlit on `http://localhost:8501`.
- `bash scripts/run_demo_stack.sh`: prints the two commands needed to run API and dashboard in separate terminals.
- `bash scripts/check_project_package.sh`: runs tests, ruff, format check, and required artifact-file checks.

## Config Groups

| Group | Files |
| --- | --- |
| App | `configs/app/api_debug.yaml`, `configs/app/dashboard_debug.yaml` |
| Data | `configs/data/debug_sample.yaml`, `configs/data/query_generation_debug.yaml` |
| Retrieval | `configs/retrieval/bm25_debug.yaml`, `dense_debug.yaml`, `faiss_debug.yaml`, `hybrid_retrieval.yaml` |
| Recommendation | `configs/recommendation/popularity_debug.yaml`, `itemcf_debug.yaml`, `matrix_factorization_debug.yaml`, `user_history_embedding_debug.yaml` |
| Sequence | `configs/sequence/gru4rec_debug.yaml`, `sasrec_debug.yaml`, plus larger `gru4rec.yaml` and `sasrec.yaml` |
| Ranking | `configs/ranking/ranking_dataset_debug.yaml`, `lightgbm_ranker_debug.yaml`, `mlp_ranker_debug.yaml`, `cross_encoder_ranker_debug.yaml`, `mixed_ranker_debug.yaml` |
| LLM | `configs/llm/query_understanding.yaml`, `configs/llm/user_profile.yaml` |
| Text/metadata representation | `configs/multimodal/text_only_debug.yaml`, `metadata_only_debug.yaml`, `text_metadata_fusion_debug.yaml`, `multimodal_fusion_debug.yaml` |
| GenRec | `configs/genrec/direct_generator_debug.yaml`, `constrained_generator_debug.yaml`, `llm_reranker_debug.yaml`, `explanation_debug.yaml`, `genrec_debug.yaml` |
| Validation | `validation/experiments/*.yaml` |

The validation templates still use a `stage` field as a grouping/schema concept. Public documentation should explain components rather than narrate a numbered development diary.

## Data Schemas And Current Counts

| Artifact | Shape | Key columns |
| --- | --- | --- |
| `data/processed/train.parquet` | `(400, 7)` | `user_id`, `item_id`, `timestamp`, `event_type`, `rating`, `event_weight`, `split` |
| `data/processed/val.parquet` | `(50, 7)` | Same interaction schema |
| `data/processed/test.parquet` | `(50, 7)` | Same interaction schema |
| `data/processed/item_metadata.parquet` | `(100, 8)` | `item_id`, `title`, `category`, `brand`, `price`, `avg_rating`, `rating_count`, `description` |
| `data/processed/user_sequences.parquet` | `(50, 6)` | `user_id`, `item_sequence`, `timestamp_sequence`, `event_type_sequence`, `event_weight_sequence`, `sequence_length` |
| `data/processed/negative_samples.parquet` | `(2500, 4)` | `user_id`, `positive_item_id`, `negative_item_id`, `split` |
| `data/processed/query_item_pairs.parquet` | `(300, 11)` | `query_text`, `target_item_id`, `category`, `intent`, `source`, `brand`, `price_constraint`, `use_case`, `relevance_label`, `query_id`, `split` |
| `data/processed/ranking_candidates.parquet` | `(15000, 35)` | Query/candidate IDs, label, retrieval scores/ranks, lexical matches, metadata, popularity, conversion, authority, cold-start, diversity |
| `data/processed/ranking_candidates_multimodal.parquet` | `(15000, 37)` | Ranking candidates plus `multimodal_score`, `multimodal_rank` |
| `data/processed/llm_query_understanding.parquet` | `(26, 24)` | Source fields, LLM fields, rewrites, expansions, parse/schema flags, provider/model, latency/cost |
| `data/processed/user_profiles.parquet` | `(50, 21)` | Profile JSON fields, provider/model, latency/cost, behavior counts |
| `data/processed/user_profile_embeddings.parquet` | `(50, 5)` | `user_id`, `profile_text`, `embedding`, `embedding_dim`, `embedding_method` |
| `data/processed/genrec_outputs.parquet` | `(52, 17)` | Query context, method, candidate pool, recommended IDs, invalid IDs, fallback/parse/schema flags, provider/model, latency/cost |
| `data/processed/llm_rerank_outputs.parquet` | `(52, 17)` | Same output schema for rerank experiments |
| `data/processed/recommendation_explanations.parquet` | `(52, 18)` | Recommended IDs, explanations, faithfulness flags, provider/model, latency/cost |

Additional observed facts:

- Query-item pairs contain 94 unique query texts with split counts `train=243`, `val=31`, `test=26`.
- Ranking candidates contain 300 queries, 50 candidates per query, positive candidate coverage of 1.0, and no duplicate query-candidate pairs.
- GenRec and rerank output fallback rates are currently 0.0 in the generated artifacts.

## Generated Artifacts

| Path | Status |
| --- | --- |
| `data/indexes/bm25_debug.pkl` | Existing retrieval index |
| `data/indexes/dense_debug.pkl` | Existing retrieval index |
| `data/indexes/faiss_debug.pkl` | Existing retrieval index using NumPy fallback in current results |
| `data/indexes/hybrid_debug.pkl` | Existing hybrid retrieval index |
| `artifacts/models/sequence/gru4rec_debug.pt` | Existing sequence checkpoint |
| `artifacts/models/sequence/sasrec_debug.pt` | Existing sequence checkpoint |
| `artifacts/models/ranking/lightgbm_ranker_debug.pkl` | Existing ranker artifact; reported backend `numpy_linear` |
| `artifacts/models/ranking/mlp_ranker_debug.pt` | Existing ranker artifact |
| `artifacts/models/ranking/cross_encoder_ranker_debug.pt` | Existing ranker artifact |
| `artifacts/models/ranking/mixed_ranker_debug.pkl` | Existing ranker artifact |
| `data/embeddings/item_text_embeddings.parquet` | `(100, 8)`, method `text_only`, dimension 237, no image signal |
| `data/embeddings/item_metadata_embeddings.parquet` | `(100, 8)`, method `metadata_only`, dimension 17, no image signal |
| `data/embeddings/item_multimodal_embeddings.parquet` | `(100, 8)`, method `text_metadata_fusion`, dimension 254, no image signal |

## Validation Result Schemas

| File | Shape | Columns |
| --- | --- | --- |
| `validation/results/retrieval_results.csv` | `(4, 15)` | `method`, `split`, `num_queries`, `top_k`, `recall_at_10`, `recall_at_20`, `recall_at_50`, `mrr_at_10`, `mrr_at_20`, `mrr_at_50`, `query_coverage`, `avg_latency_ms`, `p95_latency_ms`, `index_backend`, `config_path` |
| `validation/results/recommender_baselines.csv` | `(4, 17)` | `method`, `split`, `num_users`, `top_k`, HitRate/Recall/NDCG/MRR at 10/20, coverage at 10/20, latency, `config_path` |
| `validation/results/sequence_results.csv` | `(2, 18)` | Same recommender-style metrics plus `checkpoint_path` |
| `validation/results/ranking_results.csv` | `(4, 19)` | `method`, `backend`, `split`, `num_queries`, `top_k`, NDCG/MRR/Precision/Recall at 10/20, `auc`, `candidate_coverage`, latency, `config_path`, `model_path` |
| `validation/results/llm_query_understanding_results.csv` | `(1, 18)` | `stage`, `method`, `provider`, `model`, `split`, `num_queries`, parse/schema/match rates, expanded query count, latency, cost, `config_path` |
| `validation/results/user_profile_results.csv` | `(1, 14)` | `stage`, `method`, `provider`, `model`, `num_users`, generation/schema/nonempty/embedding metrics, profile length, latency, cost, `config_path` |
| `validation/results/multimodal_results.csv` | `(4, 27)` | `stage`, `method`, `split`, query/item counts, embedding dim, Recall/NDCG/MRR, cold-start, long-tail, coverage, diversity, image availability, latency, `config_path` |
| `validation/results/genrec_results.csv` | `(12, 19)` | `stage`, `method`, `provider`, `model`, `split`, `num_queries`, validity/hallucination/parse/schema/fallback metrics, ranking metrics, explanation faithfulness, latency, cost, `config_path` |
| `validation/results/final_leaderboard.csv` | `(31, 16)` | Shared normalized leaderboard fields: `stage`, `method`, config/split/examples, primary/secondary/coverage metrics, latency, backend, source path, notes |

## Current Selected Local Results

| Component | Selected row | Key metric |
| --- | --- | --- |
| Retrieval | `bm25`, backend `local` | Recall@50 1.000000; MRR@10 0.228785 |
| Recommendation | `itemcf` | NDCG@10 0.055705; Recall@10 0.120000; Coverage@10 0.980000 |
| Sequential recommendation | `sasrec` | NDCG@10 0.029743; MRR@10 0.020000; Coverage@10 0.670000 |
| Ranking | `lightgbm`, backend `numpy_linear` | NDCG@10 0.299915; MRR@10 0.227442; Recall@10 0.538462 |
| LLM query understanding | `llm_query_understanding`, provider `mock` | Schema-valid rate 1.000000; intent-match rate 1.000000; category-match rate 0.923077 |
| User profiles | `llm_user_profile`, provider `mock` | Generation success 1.000000; embedding coverage 1.000000 |
| Text/metadata representation | `text_metadata_fusion` | NDCG@10 0.292996; Recall@10 0.538462; cold-start Recall@10 0.444444 |
| GenRec safety | `candidate_constrained_generation`, provider `mock` | Valid item rate 1.000000; hallucination rate 0.000000; NDCG@10 0.245409 |

## API Endpoint Map

Configured via `app/api/main.py` and `app/api/routes.py`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Root message with docs and health links |
| `GET` | `/health` | Service and artifact health |
| `GET` | `/artifacts/status` | Artifact existence table |
| `POST` | `/search` | Local search/ranking demo |
| `GET` | `/recommend/{user_id}` | Local user recommendation demo |
| `POST` | `/llm/query-understanding` | Mock structured query understanding |
| `POST` | `/genrec` | Candidate-constrained GenRec demo |
| `GET` | `/models/leaderboard` | Final leaderboard |
| `GET` | `/metrics/business` | Synthetic proxy business metrics |
| `GET` | `/errors/taxonomy` | Error taxonomy markdown |

Request/response schemas are defined in `app/api/schemas.py`.

## Dashboard Page Map

| File | Page |
| --- | --- |
| `app/dashboard/streamlit_app.py` | Landing page with artifact status |
| `app/dashboard/pages/1_Search_Demo.py` | Search demo |
| `app/dashboard/pages/2_LLM_Query_Understanding.py` | Structured mock query understanding |
| `app/dashboard/pages/3_User_Profile.py` | User profile summaries |
| `app/dashboard/pages/4_Ranking_Pipeline.py` | Ranking candidates and metrics |
| `app/dashboard/pages/5_GenRec_Demo.py` | Candidate-constrained GenRec and explanations |
| `app/dashboard/pages/6_Model_Comparison.py` | Final leaderboard and selected rows |
| `app/dashboard/pages/7_Error_Analysis.py` | Error taxonomy |
| `app/dashboard/pages/8_Business_Metrics.py` | Synthetic proxy metrics |

Shared helpers live in `app/dashboard/components.py`, `app/dashboard/data_loader.py`, and `app/dashboard/demo_helpers.py`.

## Runtime Dependencies

Package metadata requires Python `>=3.10`.

Core dependencies:

- `numpy`
- `pandas`
- `pyyaml`
- `python-dotenv`
- `pyarrow`
- `torch`
- `fastapi`
- `uvicorn`
- `streamlit`
- `requests`
- `httpx`

Development dependencies:

- `pytest`
- `ruff`

## Optional Dependencies And Backends

| Optional capability | Current environment | Fallback/default |
| --- | --- | --- |
| FAISS vector search | `faiss` unavailable | Exact NumPy similarity; retrieval result backend `numpy` |
| LightGBM ranking | `lightgbm` unavailable | `NumpyLinearRanker`; ranking result backend `numpy_linear` |
| OpenAI LLM client | `openai` unavailable and external calls disabled by config | `MockLLMClient` |
| Local HuggingFace client | `transformers` unavailable by default | `MockLLMClient` |
| Image embeddings | No real image data; `image_available_rate=0.0` | Text and metadata encoders only |

## Fallback Behavior

- Retrieval falls back from FAISS-style vector search to NumPy exact search.
- The LightGBM-compatible ranker falls back to a NumPy linear ranker when LightGBM is absent.
- LLM query understanding and user profiles use deterministic mock clients by default.
- Query-understanding rows fall back to deterministic rule parsing on parse/schema errors.
- Candidate-constrained GenRec validates output item IDs against candidates and can fall back to the original ranked candidates.
- Explanation generation can use deterministic templates and rule-based faithfulness checks.
- API and dashboard show artifact status and command hints for missing files.

## Known Limitations

- Current experiments use synthetic debug data.
- The default LLM path is deterministic and mocked.
- Optional OpenAI and local-HuggingFace wrappers are not required for the default workflow.
- The current content representation uses text and structured metadata; no real image embeddings are present.
- The reported LightGBM-style ranking result used the `numpy_linear` backend in this environment.
- The reported FAISS-style retrieval result used the `numpy` backend in this environment.
- API and dashboard components are local demos, not a production deployment.
- Business metrics are synthetic proxies, not real GMV, revenue, CTR, or CVR impact.
- Query ranking is primarily query-item relevance/content/proxy ranking because the synthetic query benchmark lacks real query-user logs.
- GenRec metrics validate catalog constraints, parsing, fallback, and hallucination-control mechanics under a mock client; they are not production LLM quality results.
