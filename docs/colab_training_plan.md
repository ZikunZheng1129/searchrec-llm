# Colab Training Plan

Colab is not required for the current local demo. The current project is intentionally runnable on a laptop with synthetic debug data and mock LLM outputs.

## When To Use Colab

Use Colab or a GPU environment for:

- Larger SASRec and GRU4Rec training.
- BERT4Rec experiments.
- Two-tower neural retrieval.
- Cross-encoder training.
- Large-scale embedding generation.
- CLIP or image-embedding experiments.
- Public real-data experiments such as Amazon Reviews 2023.

## Example Colab Setup

```bash
git clone <repo>
cd SearchRec-LLM
pip install -r requirements.txt
pip install -e .
python src/pipelines/build_dataset.py --config configs/data/debug_sample.yaml
python src/pipelines/train_sequence_model.py --config configs/sequence/sasrec_debug.yaml
```

## GPU Checklist

- Confirm `torch.cuda.is_available()`.
- Use a larger data config only after the debug config passes.
- Log exact config paths and seeds.
- Save checkpoints under `artifacts/models/`.
- Export result CSVs under `validation/results/`.
- Regenerate `validation/results/final_leaderboard.csv`.

## Artifact Sync Strategy

- Keep configs and source code in Git.
- Keep large model checkpoints and embeddings outside Git unless they are intentionally small debug artifacts.
- Store final result CSVs and reports for comparison.
- Use clear names such as `sasrec_amazon_reviews_v1.yaml` and `sequence_results_amazon_reviews.csv`.

## Experiment Tracking Strategy

- Add a YAML template under `validation/experiments/`.
- Write result CSVs with the standard columns documented in `docs/experiment_tracking.md`.
- Regenerate final reports with `src/pipelines/generate_validation_report.py`.
- Note whether results were produced locally, on Colab, or on another GPU environment.

Colab is a future scaling path, not a requirement for the current local portfolio system.
