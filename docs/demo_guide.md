# Demo Guide

## Purpose

SearchRec-LLM is demoable through a local FastAPI service and a Streamlit dashboard.
The demo reads existing local artifacts from the data, retrieval, ranking, LLM, text/metadata, GenRec, and validation workflows.
It does not train models on startup and does not call external APIs by default.

## Required Artifacts

Run the required debug pipelines before the full demo:

```bash
python3 src/pipelines/build_dataset.py --config configs/data/debug_sample.yaml
python3 src/pipelines/generate_queries.py --config configs/data/query_generation_debug.yaml
python3 src/pipelines/build_ranking_dataset.py --config configs/ranking/ranking_dataset_debug.yaml
python3 src/pipelines/augment_ranking_with_multimodal.py --config configs/multimodal/multimodal_fusion_debug.yaml
python3 src/pipelines/evaluate_llm_query_understanding.py --config configs/llm/query_understanding.yaml
python3 src/pipelines/generate_user_profiles.py --config configs/llm/user_profile.yaml
python3 src/pipelines/evaluate_genrec.py --config configs/genrec/genrec_debug.yaml
python3 src/pipelines/generate_validation_report.py --config validation/experiments/stage5_debug_validation.yaml
```

## Run The API

```bash
bash scripts/launch_api.sh
```

Open `http://127.0.0.1:8000/docs`.

## Run The Dashboard

```bash
bash scripts/launch_dashboard.sh
```

Open `http://localhost:8501`.

## Demo Flow

1. Query understanding with the mock LLM.
2. Search and ranking candidates.
3. Candidate-constrained GenRec.
4. Evidence explanations.
5. Model comparison from validation CSVs.
6. Error taxonomy and synthetic proxy metrics.

## Troubleshooting

If an artifact is missing, the API and dashboard show the file path and a
pipeline command to regenerate it.

## Caveat

All current data is synthetic debug data. The demo is for local system behavior
and interview walkthroughs, not production performance claims.
