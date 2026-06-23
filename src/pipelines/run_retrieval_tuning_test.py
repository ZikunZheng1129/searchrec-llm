"""Run a small retrieval tuning comparison against a fixed benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_eval import (  # noqa: E402
    evaluate_retriever,
    summarize_retrieval_metrics,
)
from src.pipelines.build_index import (  # noqa: E402
    load_retriever,
    require_input_paths,
    resolve_path,
    run_build_index,  # noqa: E402
)
from src.pipelines.evaluate_retrieval import run_evaluate_retrieval  # noqa: E402
from src.utils.config import load_yaml_config, resolve_project_path  # noqa: E402
from src.utils.io import read_parquet  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

DEFAULT_BENCHMARK_CONFIG = Path("configs/retrieval/retrieval_tuning_benchmark.yaml")
DEFAULT_CANDIDATE_CONFIGS = [
    Path("configs/retrieval/retrieval_tuning_bm25_mrr_optimized.yaml"),
    Path("configs/retrieval/retrieval_tuning_bm25_field_weighted.yaml"),
    Path("configs/retrieval/retrieval_tuning_query_aware_boost.yaml"),
]
DEFAULT_OUTPUT_PATH = Path("validation/results/retrieval_tuning_comparison.csv")
DEFAULT_LOG_PATH = Path("validation/reports/retrieval_tuning_log.md")
DEFAULT_PER_QUERY_PATH = Path("validation/results/retrieval_tuning_per_query.csv")
DEFAULT_BREAKDOWN_PATH = Path("validation/results/retrieval_tuning_query_type_breakdown.csv")
DEFAULT_STRICT_PATH = Path("validation/results/retrieval_tuning_strict_comparison.csv")
STRICT_TOP_K = 10
STRICT_K_VALUES = [3, 5, 10]
COMPARISON_METRICS = [
    "recall_at_5",
    "recall_at_10",
    "recall_at_20",
    "recall_at_50",
    "mrr_at_5",
    "mrr_at_10",
    "mrr_at_20",
    "mrr_at_50",
    "avg_latency_ms",
    "p95_latency_ms",
]


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolve_project_path(str(candidate))


def build_comparison_rows(
    benchmark: dict[str, Any],
    candidates: list[dict[str, Any]],
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Build candidate-versus-benchmark comparison rows."""
    selected_metrics = metrics or COMPARISON_METRICS
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row: dict[str, Any] = {
            "benchmark_method": benchmark.get("method", ""),
            "candidate_method": candidate.get("method", ""),
            "benchmark_config_path": benchmark.get("config_path", ""),
            "candidate_config_path": candidate.get("config_path", ""),
            "split": candidate.get("split", benchmark.get("split", "")),
            "num_queries": candidate.get("num_queries", benchmark.get("num_queries", 0)),
            "top_k": candidate.get("top_k", benchmark.get("top_k", 0)),
        }
        for metric in selected_metrics:
            benchmark_value = float(benchmark.get(metric, 0.0))
            candidate_value = float(candidate.get(metric, 0.0))
            row[f"benchmark_{metric}"] = benchmark_value
            row[f"candidate_{metric}"] = candidate_value
            row[f"delta_{metric}"] = candidate_value - benchmark_value
        rows.append(row)
    return pd.DataFrame(rows)


def target_rank(retrieved_item_ids: list[Any], target_item_id: str) -> int:
    """Return 1-based target rank, or 0 when the target was not retrieved."""
    target = str(target_item_id)
    for rank, item_id in enumerate(retrieved_item_ids, start=1):
        if str(item_id) == target:
            return rank
    return 0


def build_query_type_breakdown(per_query: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-query retrieval metrics by config and query type."""
    if per_query.empty:
        return pd.DataFrame()

    metric_columns = [
        column
        for column in per_query.columns
        if column.startswith("recall_at_") or column.startswith("mrr_at_")
    ]
    rows = []
    group_columns = ["config_name", "query_type"]
    for group_values, group in per_query.groupby(group_columns, dropna=False):
        config_name, query_type = group_values
        row: dict[str, Any] = {
            "config_name": config_name,
            "query_type": query_type,
            "num_queries": int(len(group)),
            "avg_target_rank": float(group["target_rank"].replace(0, pd.NA).dropna().mean()),
            "miss_rate_at_10": float(1.0 - group["hit_at_10"].mean()),
        }
        for column in metric_columns:
            row[column] = float(group[column].mean())
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["config_name", "query_type"]).reset_index(drop=True)


def _experiment_metadata(config_path: Path) -> dict[str, str]:
    config = load_yaml_config(config_path)
    experiment = config.get("experiment", {})
    return {
        "name": str(experiment.get("name", config_path.stem)),
        "change": str(experiment.get("change", "")),
        "config_path": str(config_path),
    }


def _format_config_value(value: Any) -> str:
    if value in (None, "", [], {}):
        return "none"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(f"{key}={value[key]}" for key in sorted(value))
    return str(value)


def _retrieval_config_summary(config_path: Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    retrieval = config.get("retrieval", {})
    bm25 = config.get("bm25", {})
    boosts = config.get("boosts", {})
    evaluation = config.get("evaluation", {})
    return {
        "method": retrieval.get("method", ""),
        "top_k": retrieval.get("top_k", ""),
        "item_text_fields": retrieval.get("item_text_fields", []),
        "k1": bm25.get("k1", ""),
        "b": bm25.get("b", ""),
        "field_weights": bm25.get("field_weights", {}),
        "boosts": {key: value for key, value in boosts.items() if key != "use_case_terms"},
        "use_case_terms": boosts.get("use_case_terms", {}),
        "k_values": evaluation.get("k_values", []),
    }


def _changed_from_previous(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
) -> str:
    if previous is None:
        return "benchmark baseline"
    changed = [
        key
        for key in [
            "method",
            "item_text_fields",
            "k1",
            "b",
            "field_weights",
            "boosts",
            "use_case_terms",
            "top_k",
            "k_values",
        ]
        if current.get(key) != previous.get(key)
    ]
    return ", ".join(changed) if changed else "no parameter change"


def _build_change_rows(
    config_paths: list[Path],
    metadata: list[dict[str, str]],
) -> list[dict[str, Any]]:
    config_summaries = [_retrieval_config_summary(config_path) for config_path in config_paths]
    rows = []
    previous: dict[str, Any] | None = None
    for step_index, (config_summary, meta) in enumerate(
        zip(config_summaries, metadata, strict=True)
    ):
        rows.append(
            {
                "step": step_index,
                "name": meta["name"],
                "changed_from_previous": _changed_from_previous(config_summary, previous),
                "item_text_fields": _format_config_value(config_summary["item_text_fields"]),
                "k1": config_summary["k1"],
                "b": config_summary["b"],
                "field_weights": _format_config_value(config_summary["field_weights"]),
                "boosts": _format_config_value(config_summary["boosts"]),
                "top_k": config_summary["top_k"],
                "k_values": _format_config_value(config_summary["k_values"]),
            }
        )
        previous = config_summary
    return rows


def _load_query_pairs_for_config(config_path: Path) -> pd.DataFrame:
    config = load_yaml_config(config_path)
    input_paths = require_input_paths(config.get("input", {}))
    query_item_pairs = read_parquet(input_paths["query_item_pairs_path"])
    split = str(config.get("evaluation", {}).get("split", "test"))
    return query_item_pairs[query_item_pairs["split"] == split].reset_index(drop=True)


def _evaluate_config_per_query(
    config_path: Path,
    config_name: str,
    top_k: int,
    k_values: list[int],
) -> pd.DataFrame:
    config = load_yaml_config(config_path)
    method = str(config.get("retrieval", {}).get("method"))
    index_path = resolve_path(config.get("output", {})["index_path"])
    retriever = load_retriever(method, index_path)
    query_item_pairs = _load_query_pairs_for_config(config_path)
    per_query = evaluate_retriever(
        retriever=retriever,
        query_item_pairs=query_item_pairs,
        k_values=k_values,
        top_k=top_k,
    )

    metadata_columns = [
        "query_id",
        "source",
        "intent",
        "category",
        "brand",
        "price_constraint",
        "use_case",
    ]
    available_columns = [
        column for column in metadata_columns if column in query_item_pairs.columns
    ]
    per_query = per_query.merge(
        query_item_pairs[available_columns],
        on="query_id",
        how="left",
    )
    per_query["config_name"] = config_name
    per_query["query_type"] = per_query.get("source", "unknown").fillna("unknown").astype(str)
    per_query["target_rank"] = per_query.apply(
        lambda row: target_rank(row["retrieved_item_ids"], row["target_item_id"]),
        axis=1,
    )
    per_query["hit_at_5"] = (per_query["target_rank"].between(1, 5)).astype(int)
    per_query["hit_at_10"] = (per_query["target_rank"].between(1, 10)).astype(int)
    per_query["top_5_item_ids"] = per_query["retrieved_item_ids"].apply(
        lambda item_ids: ", ".join(str(item_id) for item_id in item_ids[:5])
    )
    return per_query


def _build_strict_comparison(
    benchmark_summary: dict[str, Any],
    candidate_summaries: list[dict[str, Any]],
) -> pd.DataFrame:
    return build_comparison_rows(
        benchmark_summary,
        candidate_summaries,
        metrics=[
            "recall_at_3",
            "recall_at_5",
            "recall_at_10",
            "mrr_at_3",
            "mrr_at_5",
            "mrr_at_10",
        ],
    )


def _build_error_examples(
    per_query: pd.DataFrame,
    benchmark_name: str,
    candidate_name: str,
    limit: int = 5,
) -> dict[str, pd.DataFrame]:
    columns = [
        "query_text",
        "query_type",
        "target_item_id",
        "benchmark_rank",
        "candidate_rank",
        "rank_delta",
        "top_5_item_ids",
    ]
    benchmark = per_query[per_query["config_name"] == benchmark_name][
        ["query_id", "target_rank"]
    ].rename(columns={"target_rank": "benchmark_rank"})
    candidate = per_query[per_query["config_name"] == candidate_name].copy()
    joined = candidate.merge(benchmark, on="query_id", how="left")
    joined = joined.rename(columns={"target_rank": "candidate_rank"})
    benchmark_effective_rank = joined["benchmark_rank"].where(joined["benchmark_rank"] > 0, 999)
    candidate_effective_rank = joined["candidate_rank"].where(joined["candidate_rank"] > 0, 999)
    joined["rank_delta"] = benchmark_effective_rank - candidate_effective_rank

    wins = joined[
        ((joined["benchmark_rank"] == 0) | (joined["benchmark_rank"] > 10))
        & joined["candidate_rank"].between(1, 10)
    ].sort_values(["candidate_rank", "query_id"])
    losses = joined[
        joined["benchmark_rank"].between(1, 10)
        & ((joined["candidate_rank"] == 0) | (joined["candidate_rank"] > 10))
    ].sort_values(["benchmark_rank", "query_id"])
    rank_gains = joined[joined["candidate_rank"] > 0].sort_values(
        ["rank_delta", "query_id"],
        ascending=[False, True],
    )
    rank_drops = joined[joined["benchmark_rank"] > 0].sort_values(
        ["rank_delta", "query_id"],
        ascending=[True, True],
    )
    return {
        "wins": wins[columns].head(limit),
        "losses": losses[columns].head(limit),
        "rank_gains": rank_gains[columns].head(limit),
        "rank_drops": rank_drops[columns].head(limit),
    }


def _format_metric(value: Any) -> str:
    if pd.isna(value):
        return "n/a"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [_format_metric(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_experiment_log(
    benchmark: dict[str, Any],
    candidates: list[dict[str, Any]],
    benchmark_config: Path,
    candidate_configs: list[Path],
    comparison: pd.DataFrame,
    strict_comparison: pd.DataFrame,
    per_query: pd.DataFrame,
    breakdown: pd.DataFrame,
) -> str:
    """Build a Markdown log for the retrieval tuning run."""
    summaries = [benchmark, *candidates]
    metadata = [
        _experiment_metadata(benchmark_config),
        *[_experiment_metadata(candidate_config) for candidate_config in candidate_configs],
    ]
    config_paths = [benchmark_config, *candidate_configs]
    change_rows = _build_change_rows(config_paths, metadata)
    result_rows = []
    for step_index, (summary, meta) in enumerate(zip(summaries, metadata, strict=True)):
        result_rows.append(
            {
                "step": step_index,
                "name": meta["name"],
                "change": meta["change"],
                "recall_at_5": summary.get("recall_at_5", 0.0),
                "recall_at_10": summary.get("recall_at_10", 0.0),
                "recall_at_50": summary.get("recall_at_50", 0.0),
                "mrr_at_5": summary.get("mrr_at_5", 0.0),
                "mrr_at_10": summary.get("mrr_at_10", 0.0),
                "mrr_at_50": summary.get("mrr_at_50", 0.0),
                "avg_latency_ms": summary.get("avg_latency_ms", 0.0),
            }
        )

    delta_rows = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        previous = summaries[candidate_index - 1]
        meta = metadata[candidate_index]
        row = {
            "step": candidate_index,
            "name": meta["name"],
            "delta_vs_previous_mrr_at_10": float(candidate.get("mrr_at_10", 0.0))
            - float(previous.get("mrr_at_10", 0.0)),
            "delta_vs_benchmark_mrr_at_10": comparison.iloc[candidate_index - 1]["delta_mrr_at_10"],
            "delta_vs_benchmark_recall_at_10": comparison.iloc[candidate_index - 1][
                "delta_recall_at_10"
            ],
            "delta_vs_benchmark_recall_at_50": comparison.iloc[candidate_index - 1][
                "delta_recall_at_50"
            ],
        }
        delta_rows.append(row)

    strict_rows = strict_comparison.to_dict("records")
    breakdown_rows = breakdown.to_dict("records")
    benchmark_name = metadata[0]["name"]
    error_sections = []
    for candidate_meta in metadata[1:]:
        examples = _build_error_examples(per_query, benchmark_name, candidate_meta["name"])
        error_sections.extend(
            [
                f"### {candidate_meta['name']} Wins At 10",
                _markdown_table(
                    examples["wins"].to_dict("records"),
                    [
                        "query_text",
                        "query_type",
                        "target_item_id",
                        "benchmark_rank",
                        "candidate_rank",
                        "rank_delta",
                    ],
                ),
                f"### {candidate_meta['name']} Losses At 10",
                _markdown_table(
                    examples["losses"].to_dict("records"),
                    [
                        "query_text",
                        "query_type",
                        "target_item_id",
                        "benchmark_rank",
                        "candidate_rank",
                        "rank_delta",
                    ],
                ),
            ]
        )

    return (
        "\n\n".join(
            [
                "# Retrieval Tuning Log",
                "This log compares each retrieval tuning step against the fixed BM25 benchmark "
                "using the current synthetic debug data.",
                "## Step Changes",
                _markdown_table(
                    change_rows,
                    [
                        "step",
                        "name",
                        "changed_from_previous",
                        "item_text_fields",
                        "k1",
                        "b",
                        "field_weights",
                        "boosts",
                        "top_k",
                        "k_values",
                    ],
                ),
                "## Step Results",
                _markdown_table(
                    result_rows,
                    [
                        "step",
                        "name",
                        "change",
                        "recall_at_5",
                        "recall_at_10",
                        "recall_at_50",
                        "mrr_at_5",
                        "mrr_at_10",
                        "mrr_at_50",
                        "avg_latency_ms",
                    ],
                ),
                "## Deltas",
                _markdown_table(
                    delta_rows,
                    [
                        "step",
                        "name",
                        "delta_vs_previous_mrr_at_10",
                        "delta_vs_benchmark_mrr_at_10",
                        "delta_vs_benchmark_recall_at_10",
                        "delta_vs_benchmark_recall_at_50",
                    ],
                ),
                "## Strict Top-10 Comparison",
                _markdown_table(
                    strict_rows,
                    [
                        "candidate_config_path",
                        "delta_recall_at_3",
                        "delta_recall_at_5",
                        "delta_recall_at_10",
                        "delta_mrr_at_10",
                    ],
                ),
                "## Query Type Breakdown",
                _markdown_table(
                    breakdown_rows,
                    [
                        "config_name",
                        "query_type",
                        "num_queries",
                        "recall_at_10",
                        "mrr_at_10",
                        "avg_target_rank",
                        "miss_rate_at_10",
                    ],
                ),
                "## Error Analysis",
                "\n\n".join(error_sections),
                "## Output Artifacts",
                "- `validation/results/retrieval_tuning_results.csv`\n"
                "- `validation/results/retrieval_tuning_comparison.csv`\n"
                "- `validation/results/retrieval_tuning_strict_comparison.csv`\n"
                "- `validation/results/retrieval_tuning_per_query.csv`\n"
                "- `validation/results/retrieval_tuning_query_type_breakdown.csv`\n"
                "- `validation/reports/retrieval_tuning_log.md`",
            ]
        )
        + "\n"
    )


def _reset_results_files(config_paths: list[Path]) -> None:
    """Remove prior tuning summaries so each comparison is self-contained."""
    for config_path in config_paths:
        config = load_yaml_config(config_path)
        results_path = config.get("output", {}).get("results_path")
        if results_path:
            _resolve(results_path).unlink(missing_ok=True)


def run_retrieval_tuning_test(
    benchmark_config: str | Path = DEFAULT_BENCHMARK_CONFIG,
    candidate_configs: list[str | Path] | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    log_path: str | Path = DEFAULT_LOG_PATH,
    per_query_path: str | Path = DEFAULT_PER_QUERY_PATH,
    breakdown_path: str | Path = DEFAULT_BREAKDOWN_PATH,
    strict_path: str | Path = DEFAULT_STRICT_PATH,
) -> pd.DataFrame:
    """Build indexes, evaluate benchmark/candidates, and write a comparison CSV."""
    logger = get_logger("tiksearchrec.retrieval_tuning_test")
    candidates = candidate_configs or DEFAULT_CANDIDATE_CONFIGS

    benchmark_path = _resolve(benchmark_config)
    candidate_paths = [_resolve(candidate_config) for candidate_config in candidates]
    _reset_results_files([benchmark_path, *candidate_paths])

    logger.info("Building and evaluating benchmark: %s", benchmark_path)
    run_build_index(benchmark_path)
    benchmark_summary = run_evaluate_retrieval(benchmark_path)

    candidate_summaries = []
    for candidate_path in candidate_paths:
        logger.info("Building and evaluating candidate: %s", candidate_path)
        run_build_index(candidate_path)
        candidate_summaries.append(run_evaluate_retrieval(candidate_path))

    all_paths = [benchmark_path, *candidate_paths]
    all_metadata = [_experiment_metadata(path) for path in all_paths]
    strict_per_query_frames = [
        _evaluate_config_per_query(path, meta["name"], top_k=STRICT_TOP_K, k_values=STRICT_K_VALUES)
        for path, meta in zip(all_paths, all_metadata, strict=True)
    ]
    per_query = pd.concat(strict_per_query_frames, ignore_index=True)
    strict_summaries = [
        summarize_retrieval_metrics(frame).iloc[0].to_dict() for frame in strict_per_query_frames
    ]
    for summary, meta in zip(strict_summaries, all_metadata, strict=True):
        summary["config_path"] = meta["config_path"]
    strict_comparison = _build_strict_comparison(strict_summaries[0], strict_summaries[1:])
    breakdown = build_query_type_breakdown(per_query)

    comparison = build_comparison_rows(benchmark_summary, candidate_summaries)
    resolved_output_path = _resolve(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(resolved_output_path, index=False)
    logger.info("Retrieval tuning comparison saved to %s", resolved_output_path)

    resolved_per_query_path = _resolve(per_query_path)
    resolved_per_query_path.parent.mkdir(parents=True, exist_ok=True)
    per_query.to_csv(resolved_per_query_path, index=False)
    logger.info("Retrieval tuning per-query output saved to %s", resolved_per_query_path)

    resolved_breakdown_path = _resolve(breakdown_path)
    resolved_breakdown_path.parent.mkdir(parents=True, exist_ok=True)
    breakdown.to_csv(resolved_breakdown_path, index=False)
    logger.info("Retrieval tuning query-type breakdown saved to %s", resolved_breakdown_path)

    resolved_strict_path = _resolve(strict_path)
    resolved_strict_path.parent.mkdir(parents=True, exist_ok=True)
    strict_comparison.to_csv(resolved_strict_path, index=False)
    logger.info("Retrieval tuning strict comparison saved to %s", resolved_strict_path)

    resolved_log_path = _resolve(log_path)
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_log_path.write_text(
        build_experiment_log(
            benchmark_summary,
            candidate_summaries,
            benchmark_path,
            candidate_paths,
            comparison,
            strict_comparison,
            per_query,
            breakdown,
        ),
        encoding="utf-8",
    )
    logger.info("Retrieval tuning log saved to %s", resolved_log_path)
    return comparison


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run a small retrieval tuning A/B test.")
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=DEFAULT_BENCHMARK_CONFIG,
        help="Benchmark retrieval config path.",
    )
    parser.add_argument(
        "--candidate-config",
        type=Path,
        action="append",
        dest="candidate_configs",
        help="Candidate retrieval config path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Comparison CSV output path.",
    )
    parser.add_argument(
        "--log-output",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Markdown experiment log output path.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_retrieval_tuning_test(
        benchmark_config=args.benchmark_config,
        candidate_configs=args.candidate_configs,
        output_path=args.output,
        log_path=args.log_output,
    )


if __name__ == "__main__":
    main()
