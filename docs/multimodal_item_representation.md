# Text And Metadata Item Representation

The content-representation layer adds local item vectors for discovery. For the current
synthetic catalog, the real available signals are item text and metadata. Image
support is an interface for future precomputed features, not a real vision model
in the reported experiments.

## Why It Matters

Large-scale discovery systems need strong content representations for cold-start
and long-tail items. Text and metadata can help retrieve candidates even when
behavioral data is sparse.

## Local Scope

- `TextItemEncoder`: local TF-IDF over title, category, brand, and description.
- `MetadataItemEncoder`: one-hot category/brand plus normalized numeric fields.
- `ImageItemEncoder`: optional interface for future image paths or precomputed
  embeddings.
- `MultimodalFusionEncoder`: weighted concatenation of available components.
- `MultimodalRetriever`: exact local vector search over item embeddings.

## Image Caveat

The synthetic debug dataset has no real image data. The current workflow does not download
images, CLIP, OpenCLIP, or VLM models. If no image embeddings are present, image
features are skipped and `image_available_rate=0.0`.

## Ranking Augmentation

`augment_ranking_with_multimodal.py` adds `multimodal_score` and
`multimodal_rank` to ranking candidates while preserving row count.

## Metrics

- Recall@K, NDCG@K, MRR@K
- cold-start recall@K
- long-tail coverage@K
- catalog coverage@K
- category diversity@K
- latency

Category diversity is the average unique categories in top-K normalized by the
smaller of K and the number of available categories.

## Run

```bash
python src/pipelines/build_multimodal_embeddings.py --config configs/multimodal/text_metadata_fusion_debug.yaml
python src/pipelines/evaluate_multimodal.py --config configs/multimodal/text_metadata_fusion_debug.yaml
python src/pipelines/augment_ranking_with_multimodal.py --config configs/multimodal/multimodal_fusion_debug.yaml
python src/pipelines/generate_validation_report.py --config validation/experiments/stage5_debug_validation.yaml
```

## Caveats

- Synthetic data only.
- No real image signal yet.
- Cold-start and long-tail slices are simulated from low interaction counts.
- No production cold-start claim.
- No VLM or learned fusion yet.

## Future Work

- CLIP or precomputed image embeddings.
- VLM captions.
- Learned fusion.
- Multimodal ranking features.
- Real catalog cold-start evaluation.
