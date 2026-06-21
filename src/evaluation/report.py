"""Stage 5 validation report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

LEADERBOARD_COLUMNS = [
    "stage",
    "method",
    "config_path",
    "split",
    "num_eval_examples",
    "primary_metric_name",
    "primary_score",
    "secondary_metric_name",
    "secondary_score",
    "coverage_metric_name",
    "coverage_score",
    "avg_latency_ms",
    "p95_latency_ms",
    "index_backend",
    "source_results_path",
    "notes",
]


def read_results_csv(path: str | Path, required: bool = True) -> pd.DataFrame:
    """Read a result CSV, optionally ignoring missing future-stage files."""
    result_path = Path(path)
    if not result_path.exists():
        if required:
            raise FileNotFoundError(f"Required result file not found: {result_path}")
        return pd.DataFrame()

    df = pd.read_csv(result_path)
    if df.empty and required:
        raise ValueError(f"Required result file is empty: {result_path}")
    return df


def _value(row: pd.Series, column: str, default: Any = "") -> Any:
    return row[column] if column in row and pd.notna(row[column]) else default


def _normalize_float(value: Any, default: float = 0.0) -> float:
    return float(value) if pd.notna(value) and value != "" else default


def normalize_retrieval_results(
    df: pd.DataFrame,
    source_results_path: str | Path = "validation/results/retrieval_results.csv",
) -> pd.DataFrame:
    """Normalize Stage 3 retrieval results into the final leaderboard schema."""
    if df.empty:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)

    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "stage": "retrieval",
                "method": str(_value(row, "method")),
                "config_path": str(_value(row, "config_path")),
                "split": str(_value(row, "split")),
                "num_eval_examples": int(_normalize_float(_value(row, "num_queries", 0))),
                "primary_metric_name": "recall_at_50",
                "primary_score": _normalize_float(_value(row, "recall_at_50", 0.0)),
                "secondary_metric_name": "mrr_at_10",
                "secondary_score": _normalize_float(_value(row, "mrr_at_10", 0.0)),
                "coverage_metric_name": "query_coverage",
                "coverage_score": _normalize_float(_value(row, "query_coverage", 0.0)),
                "avg_latency_ms": _normalize_float(_value(row, "avg_latency_ms", 0.0)),
                "p95_latency_ms": _normalize_float(_value(row, "p95_latency_ms", 0.0)),
                "index_backend": str(_value(row, "index_backend", "local")),
                "source_results_path": str(source_results_path),
                "notes": "Stage 3 synthetic query-item retrieval result.",
            }
        )

    return pd.DataFrame(rows, columns=LEADERBOARD_COLUMNS)


def normalize_recommendation_results(
    df: pd.DataFrame,
    source_results_path: str | Path = "validation/results/recommender_baselines.csv",
) -> pd.DataFrame:
    """Normalize Stage 4 recommendation results into the final leaderboard schema."""
    if df.empty:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)

    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "stage": "recommendation",
                "method": str(_value(row, "method")),
                "config_path": str(_value(row, "config_path")),
                "split": str(_value(row, "split")),
                "num_eval_examples": int(_normalize_float(_value(row, "num_users", 0))),
                "primary_metric_name": "ndcg_at_10",
                "primary_score": _normalize_float(_value(row, "ndcg_at_10", 0.0)),
                "secondary_metric_name": "recall_at_10",
                "secondary_score": _normalize_float(_value(row, "recall_at_10", 0.0)),
                "coverage_metric_name": "coverage_at_10",
                "coverage_score": _normalize_float(_value(row, "coverage_at_10", 0.0)),
                "avg_latency_ms": _normalize_float(_value(row, "avg_latency_ms", 0.0)),
                "p95_latency_ms": _normalize_float(_value(row, "p95_latency_ms", 0.0)),
                "index_backend": "n/a",
                "source_results_path": str(source_results_path),
                "notes": "Stage 4 synthetic debug recommendation result.",
            }
        )

    return pd.DataFrame(rows, columns=LEADERBOARD_COLUMNS)


def normalize_sequence_results(
    df: pd.DataFrame,
    source_results_path: str | Path = "validation/results/sequence_results.csv",
) -> pd.DataFrame:
    """Normalize Stage 6 sequence results into the final leaderboard schema."""
    if df.empty:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)

    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "stage": "sequence",
                "method": str(_value(row, "method")),
                "config_path": str(_value(row, "config_path")),
                "split": str(_value(row, "split")),
                "num_eval_examples": int(_normalize_float(_value(row, "num_users", 0))),
                "primary_metric_name": "ndcg_at_10",
                "primary_score": _normalize_float(_value(row, "ndcg_at_10", 0.0)),
                "secondary_metric_name": "recall_at_10",
                "secondary_score": _normalize_float(_value(row, "recall_at_10", 0.0)),
                "coverage_metric_name": "coverage_at_10",
                "coverage_score": _normalize_float(_value(row, "coverage_at_10", 0.0)),
                "avg_latency_ms": _normalize_float(_value(row, "avg_latency_ms", 0.0)),
                "p95_latency_ms": _normalize_float(_value(row, "p95_latency_ms", 0.0)),
                "index_backend": "n/a",
                "source_results_path": str(source_results_path),
                "notes": "Stage 6 synthetic debug sequence recommendation result.",
            }
        )

    return pd.DataFrame(rows, columns=LEADERBOARD_COLUMNS)


def normalize_ranking_results(
    df: pd.DataFrame,
    source_results_path: str | Path = "validation/results/ranking_results.csv",
) -> pd.DataFrame:
    """Normalize Stage 7 ranking results into the final leaderboard schema."""
    if df.empty:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)

    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "stage": "ranking",
                "method": str(_value(row, "method")),
                "config_path": str(_value(row, "config_path")),
                "split": str(_value(row, "split")),
                "num_eval_examples": int(_normalize_float(_value(row, "num_queries", 0))),
                "primary_metric_name": "ndcg_at_10",
                "primary_score": _normalize_float(_value(row, "ndcg_at_10", 0.0)),
                "secondary_metric_name": "mrr_at_10",
                "secondary_score": _normalize_float(_value(row, "mrr_at_10", 0.0)),
                "coverage_metric_name": "candidate_coverage",
                "coverage_score": _normalize_float(_value(row, "candidate_coverage", 0.0)),
                "avg_latency_ms": _normalize_float(_value(row, "avg_latency_ms", 0.0)),
                "p95_latency_ms": _normalize_float(_value(row, "p95_latency_ms", 0.0)),
                "index_backend": str(_value(row, "backend", "n/a")),
                "source_results_path": str(source_results_path),
                "notes": "Stage 7 synthetic debug query-item ranking result.",
            }
        )

    return pd.DataFrame(rows, columns=LEADERBOARD_COLUMNS)


def normalize_llm_query_understanding_results(
    df: pd.DataFrame,
    source_results_path: str | Path = "validation/results/llm_query_understanding_results.csv",
) -> pd.DataFrame:
    """Normalize Stage 8 LLM query-understanding results into leaderboard rows."""
    if df.empty:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)

    rows = []
    for _, row in df.iterrows():
        provider = str(_value(row, "provider", ""))
        model = str(_value(row, "model", ""))
        rows.append(
            {
                "stage": "llm_query_understanding",
                "method": str(_value(row, "method", "llm_query_understanding")),
                "config_path": str(_value(row, "config_path")),
                "split": str(_value(row, "split")),
                "num_eval_examples": int(_normalize_float(_value(row, "num_queries", 0))),
                "primary_metric_name": "schema_valid_rate",
                "primary_score": _normalize_float(_value(row, "schema_valid_rate", 0.0)),
                "secondary_metric_name": "intent_match_rate",
                "secondary_score": _normalize_float(_value(row, "intent_match_rate", 0.0)),
                "coverage_metric_name": "output_parse_success_rate",
                "coverage_score": _normalize_float(_value(row, "output_parse_success_rate", 0.0)),
                "avg_latency_ms": _normalize_float(_value(row, "avg_latency_ms", 0.0)),
                "p95_latency_ms": _normalize_float(_value(row, "p95_latency_ms", 0.0)),
                "index_backend": f"{provider}/{model}".strip("/"),
                "source_results_path": str(source_results_path),
                "notes": (
                    "Stage 8 mock/local debug LLM query-understanding result."
                    if provider == "mock"
                    else "Stage 8 optional LLM query-understanding result."
                ),
            }
        )
    return pd.DataFrame(rows, columns=LEADERBOARD_COLUMNS)


def normalize_multimodal_results(
    df: pd.DataFrame,
    source_results_path: str | Path = "validation/results/multimodal_results.csv",
) -> pd.DataFrame:
    """Normalize Stage 9 multimodal results into leaderboard rows."""
    if df.empty:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)

    rows = []
    for _, row in df.iterrows():
        has_cold_start = "cold_start_recall_at_10" in row and pd.notna(
            row["cold_start_recall_at_10"]
        )
        cold_start = _normalize_float(row["cold_start_recall_at_10"]) if has_cold_start else 0.0
        recall = _normalize_float(_value(row, "recall_at_10", 0.0))
        image_rate = _normalize_float(_value(row, "image_available_rate", 0.0))
        rows.append(
            {
                "stage": "multimodal",
                "method": str(_value(row, "method")),
                "config_path": str(_value(row, "config_path")),
                "split": str(_value(row, "split")),
                "num_eval_examples": int(_normalize_float(_value(row, "num_queries", 0))),
                "primary_metric_name": "ndcg_at_10",
                "primary_score": _normalize_float(_value(row, "ndcg_at_10", 0.0)),
                "secondary_metric_name": (
                    "cold_start_recall_at_10" if has_cold_start else "recall_at_10"
                ),
                "secondary_score": cold_start if has_cold_start else recall,
                "coverage_metric_name": "catalog_coverage_at_10",
                "coverage_score": _normalize_float(_value(row, "catalog_coverage_at_10", 0.0)),
                "avg_latency_ms": _normalize_float(_value(row, "avg_latency_ms", 0.0)),
                "p95_latency_ms": _normalize_float(_value(row, "p95_latency_ms", 0.0)),
                "index_backend": str(_value(row, "method", "local_vector")),
                "source_results_path": str(source_results_path),
                "notes": (
                    "Stage 9 local text/metadata result; no real image features available."
                    if image_rate == 0.0
                    else "Stage 9 local multimodal result with provided image features."
                ),
            }
        )
    return pd.DataFrame(rows, columns=LEADERBOARD_COLUMNS)


def normalize_genrec_results(
    df: pd.DataFrame,
    source_results_path: str | Path = "validation/results/genrec_results.csv",
) -> pd.DataFrame:
    """Normalize Stage 10 GenRec results into leaderboard rows."""
    if df.empty:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)

    rows = []
    for _, row in df.iterrows():
        provider = str(_value(row, "provider", ""))
        model = str(_value(row, "model", ""))
        method = str(_value(row, "method", ""))
        hallucination = _normalize_float(_value(row, "hallucination_rate", 0.0))
        notes = (
            "Stage 10 direct generation baseline only; not final serving design."
            if method == "direct_generation"
            else "Stage 10 candidate-grounded mock/local GenRec result."
        )
        if provider == "mock":
            notes += " Uses deterministic mock client."
        if hallucination > 0.0:
            notes += " Invalid item risk observed."
        rows.append(
            {
                "stage": "genrec",
                "method": method,
                "config_path": str(_value(row, "config_path")),
                "split": str(_value(row, "split")),
                "num_eval_examples": int(_normalize_float(_value(row, "num_queries", 0))),
                "primary_metric_name": "valid_item_rate",
                "primary_score": _normalize_float(_value(row, "valid_item_rate", 0.0)),
                "secondary_metric_name": "ndcg_at_10",
                "secondary_score": _normalize_float(_value(row, "ndcg_at_10", 0.0)),
                "coverage_metric_name": "output_parse_success_rate",
                "coverage_score": _normalize_float(_value(row, "output_parse_success_rate", 0.0)),
                "avg_latency_ms": _normalize_float(_value(row, "avg_latency_ms", 0.0)),
                "p95_latency_ms": _normalize_float(_value(row, "p95_latency_ms", 0.0)),
                "index_backend": f"{provider}/{model}".strip("/"),
                "source_results_path": str(source_results_path),
                "notes": notes,
            }
        )
    return pd.DataFrame(rows, columns=LEADERBOARD_COLUMNS)


def _sort_leaderboard(leaderboard: pd.DataFrame) -> pd.DataFrame:
    if leaderboard.empty:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    return (
        leaderboard[LEADERBOARD_COLUMNS]
        .sort_values(
            by=[
                "stage",
                "primary_score",
                "secondary_score",
                "coverage_score",
                "avg_latency_ms",
                "method",
            ],
            ascending=[True, False, False, False, True, True],
        )
        .reset_index(drop=True)
    )


def build_final_leaderboard(
    retrieval_results: pd.DataFrame | None = None,
    recommendation_results: pd.DataFrame | None = None,
    sequence_results: pd.DataFrame | None = None,
    ranking_results: pd.DataFrame | None = None,
    llm_query_understanding_results: pd.DataFrame | None = None,
    multimodal_results: pd.DataFrame | None = None,
    genrec_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine normalized current-stage result rows into one leaderboard."""
    frames = []
    if retrieval_results is not None and not retrieval_results.empty:
        if set(LEADERBOARD_COLUMNS).issubset(retrieval_results.columns):
            frames.append(retrieval_results[LEADERBOARD_COLUMNS])
        else:
            frames.append(normalize_retrieval_results(retrieval_results))
    if recommendation_results is not None and not recommendation_results.empty:
        if set(LEADERBOARD_COLUMNS).issubset(recommendation_results.columns):
            frames.append(recommendation_results[LEADERBOARD_COLUMNS])
        else:
            frames.append(normalize_recommendation_results(recommendation_results))
    if sequence_results is not None and not sequence_results.empty:
        if set(LEADERBOARD_COLUMNS).issubset(sequence_results.columns):
            frames.append(sequence_results[LEADERBOARD_COLUMNS])
        else:
            frames.append(normalize_sequence_results(sequence_results))
    if ranking_results is not None and not ranking_results.empty:
        if set(LEADERBOARD_COLUMNS).issubset(ranking_results.columns):
            frames.append(ranking_results[LEADERBOARD_COLUMNS])
        else:
            frames.append(normalize_ranking_results(ranking_results))
    if llm_query_understanding_results is not None and not llm_query_understanding_results.empty:
        if set(LEADERBOARD_COLUMNS).issubset(llm_query_understanding_results.columns):
            frames.append(llm_query_understanding_results[LEADERBOARD_COLUMNS])
        else:
            frames.append(
                normalize_llm_query_understanding_results(llm_query_understanding_results)
            )
    if multimodal_results is not None and not multimodal_results.empty:
        if set(LEADERBOARD_COLUMNS).issubset(multimodal_results.columns):
            frames.append(multimodal_results[LEADERBOARD_COLUMNS])
        else:
            frames.append(normalize_multimodal_results(multimodal_results))
    if genrec_results is not None and not genrec_results.empty:
        if set(LEADERBOARD_COLUMNS).issubset(genrec_results.columns):
            frames.append(genrec_results[LEADERBOARD_COLUMNS])
        else:
            frames.append(normalize_genrec_results(genrec_results))

    if not frames:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    return _sort_leaderboard(pd.concat(frames, ignore_index=True))


def select_best_models(leaderboard: pd.DataFrame) -> pd.DataFrame:
    """Select the best row per stage using deterministic tie-breaks."""
    if leaderboard.empty:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)

    selected_rows = []
    for _, stage_rows in leaderboard.groupby("stage", sort=True):
        sorted_stage = stage_rows.sort_values(
            by=["primary_score", "secondary_score", "coverage_score", "avg_latency_ms", "method"],
            ascending=[False, False, False, True, True],
        )
        selected_rows.append(sorted_stage.iloc[0])
    return pd.DataFrame(selected_rows, columns=LEADERBOARD_COLUMNS).reset_index(drop=True)


def write_leaderboard(leaderboard: pd.DataFrame, path: str | Path) -> None:
    """Write the final leaderboard CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _sort_leaderboard(leaderboard).to_csv(output_path, index=False)


def render_markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """Render a small DataFrame as a deterministic Markdown table."""
    if df.empty:
        return "_No rows available._"

    display = df.head(max_rows).copy() if max_rows is not None else df.copy()
    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in display.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _report_table(leaderboard: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "stage",
        "method",
        "primary_metric_name",
        "primary_score",
        "secondary_metric_name",
        "secondary_score",
        "coverage_metric_name",
        "coverage_score",
        "avg_latency_ms",
    ]
    return leaderboard[columns].copy() if not leaderboard.empty else pd.DataFrame(columns=columns)


def _best_text(best_models: pd.DataFrame, stage: str) -> str:
    stage_rows = best_models[best_models["stage"] == stage]
    if stage_rows.empty:
        return f"No {stage} baseline has been evaluated yet."
    row = stage_rows.iloc[0]
    return (
        f"`{row['method']}` selected for {stage} because it has the best "
        f"{row['primary_metric_name']}={row['primary_score']:.4f}, with "
        f"{row['secondary_metric_name']}={row['secondary_score']:.4f}, "
        f"{row['coverage_metric_name']}={row['coverage_score']:.4f}, and "
        f"avg latency {row['avg_latency_ms']:.4f} ms."
    )


def generate_model_selection_report(
    leaderboard: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Generate the Stage 5 model-selection Markdown report."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_models = select_best_models(leaderboard)
    table = render_markdown_table(_report_table(leaderboard))

    text = f"""# Final Model Selection Report

Stage 5 status: validation framework added for current local baselines.

Data caveat: all current metrics come from synthetic/local debug data.
These results are useful for model-selection mechanics and portfolio discussion,
not production claims.

## Existing Evaluated Components

- Retrieval baselines from Stage 3.
- Recommendation baselines from Stage 4.
- Sequence baselines from Stage 6, when `sequence_results.csv` exists.
- Ranking baselines from Stage 7, when `ranking_results.csv` exists.
- LLM query-understanding outputs from Stage 8, when results exist.
- Multimodal item representations from Stage 9, when `multimodal_results.csv` exists.
- GenRec, LLM reranking, and explanations from Stage 10, when `genrec_results.csv` exists.

## Final Leaderboard

{table}

## Best Retrieval Baseline

{_best_text(best_models, "retrieval")}

## Best Recommendation Baseline

{_best_text(best_models, "recommendation")}

## Best Sequence Baseline

{_best_text(best_models, "sequence")}

## Best Ranking Baseline

{_best_text(best_models, "ranking")}

## Best LLM Query-Understanding Baseline

{_best_text(best_models, "llm_query_understanding")}

## Best Multimodal Baseline

{_best_text(best_models, "multimodal")}

## Best GenRec Baseline

{_best_text(best_models, "genrec")}

## Tradeoffs

- Quality: selected separately per stage because retrieval and recommendation
  metrics are not directly comparable.
- Coverage: tracked so high quality does not hide narrow catalog exposure.
- Latency: tracked to keep local baselines honest before heavier models arrive.
- Simplicity: simple baselines are preferred when quality is tied or close.

## Not Evaluated Yet

- Real Amazon Reviews data.

## Future Validation Plan

Add future stage result CSVs into `validation/results/`, normalize them into
the same leaderboard schema, and keep model selection stage-specific.

## Interview Talking Point

“We separated unit tests from model validation. Tests prove the code works;
validation compares model choices and tradeoffs.”
"""
    output.write_text(text, encoding="utf-8")


def generate_latency_quality_report(
    leaderboard: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Generate a latency versus quality Markdown report."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    quality_columns = [
        "stage",
        "method",
        "primary_metric_name",
        "primary_score",
        "avg_latency_ms",
        "p95_latency_ms",
        "coverage_score",
    ]
    retrieval = leaderboard[leaderboard["stage"] == "retrieval"][quality_columns]
    recommendation = leaderboard[leaderboard["stage"] == "recommendation"][quality_columns]
    sequence = leaderboard[leaderboard["stage"] == "sequence"][quality_columns]
    ranking = leaderboard[leaderboard["stage"] == "ranking"][quality_columns]
    llm_query = leaderboard[leaderboard["stage"] == "llm_query_understanding"][quality_columns]
    multimodal = leaderboard[leaderboard["stage"] == "multimodal"][quality_columns]
    genrec = leaderboard[leaderboard["stage"] == "genrec"][quality_columns]

    text = f"""# Latency And Quality Tradeoff

Data caveat: all current measurements use the local synthetic debug dataset
and a small item catalog.

## Retrieval

{render_markdown_table(retrieval)}

## Recommendation

{render_markdown_table(recommendation)}

## Sequence

{render_markdown_table(sequence)}

## Ranking

{render_markdown_table(ranking)}

## LLM Query Understanding

{render_markdown_table(llm_query)}

## Multimodal

{render_markdown_table(multimodal)}

## GenRec

{render_markdown_table(genrec)}

## How To Interpret

Higher quality metrics are better, but latency and coverage make the choice
more practical. A baseline with slightly lower quality may still be valuable
when it is simpler, faster, or covers more of the catalog.

## Current Limitations

- Local synthetic dataset.
- Small item catalog.
- No production load testing yet.

## Future Work

- Larger candidate pools.
- Real FAISS indexes.
- Neural embeddings.
- Ranker latency.
- API latency.
"""
    output.write_text(text, encoding="utf-8")


def generate_validation_summary(
    leaderboard: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Generate a concise validation summary."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_models = select_best_models(leaderboard)
    generated_outputs = [
        "validation/results/final_leaderboard.csv",
        "validation/reports/final_model_selection_report.md",
        "validation/reports/latency_quality_tradeoff.md",
        "validation/reports/validation_summary.md",
    ]
    input_files = sorted(leaderboard["source_results_path"].dropna().unique().tolist())
    best_lines = [
        (
            f"- {row['stage']}: `{row['method']}` by "
            f"{row['primary_metric_name']}={row['primary_score']:.4f}"
        )
        for _, row in best_models.iterrows()
    ]

    text = f"""# Validation Summary

Current scope: Stage 5 validation over real result CSVs from synthetic/local
debug data. Ranking, LLM, and multimodal rows are included when their result
files exist.

## Input Files

{chr(10).join(f"- {path}" for path in input_files)}

## Generated Outputs

{chr(10).join(f"- {path}" for path in generated_outputs)}

## Leaderboard Rows

{len(leaderboard)}

## Best Method Per Current Stage

{chr(10).join(best_lines) if best_lines else "- No current stage rows available."}

## Missing Future Stages

- Ranking rows: included when `validation/results/ranking_results.csv` exists.
- Multimodal rows: included when `validation/results/multimodal_results.csv` exists.
- GenRec rows: included when `validation/results/genrec_results.csv` exists.

## Reproduce

```bash
python3 src/pipelines/generate_validation_report.py \\
  --config validation/experiments/stage5_debug_validation.yaml
bash scripts/run_validation.sh
```
"""
    output.write_text(text, encoding="utf-8")


def _resolve_config_path(path: str | Path, base_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate


def generate_validation_artifacts(
    config: dict[str, Any],
    config_path: str | Path | None = None,
) -> dict[str, str]:
    """Generate the Stage 5 leaderboard and Markdown reports from result CSVs."""
    base_dir = Path(config_path).resolve().parents[2] if config_path is not None else Path.cwd()
    inputs = config.get("inputs", {})
    optional_inputs = config.get("optional_inputs", {})
    outputs = config.get("outputs", {})

    retrieval_path = _resolve_config_path(inputs["retrieval_results_path"], base_dir)
    recommendation_path = _resolve_config_path(inputs["recommender_results_path"], base_dir)
    retrieval_results = normalize_retrieval_results(
        read_results_csv(retrieval_path, required=True),
        source_results_path=inputs["retrieval_results_path"],
    )
    recommendation_results = normalize_recommendation_results(
        read_results_csv(recommendation_path, required=True),
        source_results_path=inputs["recommender_results_path"],
    )
    sequence_results = pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    ranking_results = pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    llm_query_understanding_results = pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    multimodal_results = pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    genrec_results = pd.DataFrame(columns=LEADERBOARD_COLUMNS)

    for optional_name, optional_path in optional_inputs.items():
        optional_df = read_results_csv(
            _resolve_config_path(optional_path, base_dir),
            required=False,
        )
        if optional_name == "sequence_results_path" and not optional_df.empty:
            sequence_results = normalize_sequence_results(
                optional_df,
                source_results_path=optional_path,
            )
        if optional_name == "ranking_results_path" and not optional_df.empty:
            ranking_results = normalize_ranking_results(
                optional_df,
                source_results_path=optional_path,
            )
        if optional_name == "llm_query_understanding_results_path" and not optional_df.empty:
            llm_query_understanding_results = normalize_llm_query_understanding_results(
                optional_df,
                source_results_path=optional_path,
            )
        if optional_name == "multimodal_results_path" and not optional_df.empty:
            multimodal_results = normalize_multimodal_results(
                optional_df,
                source_results_path=optional_path,
            )
        if optional_name == "genrec_results_path" and not optional_df.empty:
            genrec_results = normalize_genrec_results(
                optional_df,
                source_results_path=optional_path,
            )

    leaderboard = build_final_leaderboard(
        retrieval_results,
        recommendation_results,
        sequence_results,
        ranking_results,
        llm_query_understanding_results,
        multimodal_results,
        genrec_results,
    )
    if leaderboard.empty:
        raise ValueError("No current validation rows were available for the final leaderboard")

    output_paths = {name: _resolve_config_path(path, base_dir) for name, path in outputs.items()}
    write_leaderboard(leaderboard, output_paths["final_leaderboard_path"])
    generate_model_selection_report(
        leaderboard,
        output_paths["final_model_selection_report_path"],
    )
    generate_latency_quality_report(
        leaderboard,
        output_paths["latency_quality_report_path"],
    )
    generate_validation_summary(
        leaderboard,
        output_paths["validation_summary_path"],
    )
    return {name: str(path) for name, path in output_paths.items()}
