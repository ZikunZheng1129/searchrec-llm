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
    Path("configs/retrieval/retrieval_tuning_bm25_category_heavy_no_desc.yaml"),
    Path("configs/retrieval/retrieval_tuning_bm25_light_description.yaml"),
]
DEFAULT_OUTPUT_PATH = Path("validation/results/retrieval_tuning_comparison.csv")
DEFAULT_LOG_PATH = Path("validation/reports/retrieval_tuning_log.md")
DEFAULT_PER_QUERY_PATH = Path("validation/results/retrieval_tuning_per_query.csv")
DEFAULT_BREAKDOWN_PATH = Path("validation/results/retrieval_tuning_query_type_breakdown.csv")
DEFAULT_STRICT_PATH = Path("validation/results/retrieval_tuning_strict_comparison.csv")
DEFAULT_MULTI_RELEVANT_PATH = Path("validation/results/retrieval_tuning_multi_relevant.csv")
STRICT_TOP_K = 10
STRICT_K_VALUES = [3, 5, 10]
COMPARISON_METRICS = [
    "recall_at_5",
    "recall_at_10",
    "recall_at_20",
    "recall_at_50",
    "recall_at_100",
    "mrr_at_5",
    "mrr_at_10",
    "mrr_at_20",
    "mrr_at_50",
    "mrr_at_100",
    "avg_latency_ms",
    "p95_latency_ms",
]
RETRIEVAL_PASS_METRIC = "recall_at_50"
RETRIEVAL_TIEBREAK_METRIC = "recall_at_10"


def _threshold_profile(num_items: int) -> dict[str, float]:
    """Return practical pass thresholds for the current retrieval catalog size."""
    if num_items <= 100:
        return {
            "recall_at_50": 0.90,
            "recall_at_10": 0.60,
            "hit_any_at_50": 0.90,
            "recall_multi_at_50": 0.90,
        }
    if num_items <= 1_000:
        return {
            "recall_at_50": 0.60,
            "recall_at_10": 0.30,
            "hit_any_at_50": 0.90,
            "recall_multi_at_50": 0.70,
        }
    if num_items <= 5_000:
        return {
            "recall_at_50": 0.50,
            "recall_at_10": 0.20,
            "hit_any_at_50": 0.85,
            "recall_multi_at_50": 0.60,
        }
    if num_items <= 10_000:
        return {
            "recall_at_50": 0.35,
            "recall_at_10": 0.12,
            "hit_any_at_50": 0.75,
            "recall_multi_at_50": 0.45,
        }
    return {
        "recall_at_50": 0.25,
        "recall_at_10": 0.08,
        "hit_any_at_50": 0.65,
        "recall_multi_at_50": 0.35,
    }


def _format_thresholds(thresholds: dict[str, float]) -> str:
    return (
        f"single {RETRIEVAL_PASS_METRIC}>={thresholds[RETRIEVAL_PASS_METRIC]:.2f} "
        f"or multi hit_any_at_50>={thresholds['hit_any_at_50']:.2f} "
        f"and recall_multi_at_50>={thresholds['recall_multi_at_50']:.2f}; "
        f"observe {RETRIEVAL_TIEBREAK_METRIC}>={thresholds[RETRIEVAL_TIEBREAK_METRIC]:.2f}"
    )


def _pool_pass_mode(
    summary: dict[str, Any],
    multi_relevant_row: dict[str, Any] | None,
    thresholds: dict[str, float],
) -> str:
    passes_single = (
        float(summary.get(RETRIEVAL_PASS_METRIC, 0.0)) >= thresholds[RETRIEVAL_PASS_METRIC]
    )
    if passes_single:
        return "single"
    if multi_relevant_row:
        passes_multi = (
            float(multi_relevant_row.get("hit_any_at_50", 0.0)) >= thresholds["hit_any_at_50"]
            and float(multi_relevant_row.get("recall_multi_at_50", 0.0))
            >= thresholds["recall_multi_at_50"]
        )
        if passes_multi:
            return "multi"
    return ""


def _pass_status(
    summary: dict[str, Any],
    multi_relevant_row: dict[str, Any] | None,
    thresholds: dict[str, float],
) -> str:
    pass_mode = _pool_pass_mode(summary, multi_relevant_row, thresholds)
    if pass_mode:
        text = f"pass: {pass_mode} pool"
        return f'<span style="color: green; font-weight: 600;">{text}</span>'
    failed_text = f"not pass: {RETRIEVAL_PASS_METRIC}"
    return f'<span style="color: red; font-weight: 600;">{failed_text}</span>'


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


def _first_relevant_rank(retrieved_item_ids: list[Any], relevant_item_ids: set[str], k: int) -> int:
    for rank, item_id in enumerate(retrieved_item_ids[:k], start=1):
        if str(item_id) in relevant_item_ids:
            return rank
    return 0


def build_multi_relevant_metrics(per_query: pd.DataFrame, k_values: list[int]) -> pd.DataFrame:
    """Evaluate retrieval with all targets for the same query text as relevant."""
    if per_query.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_columns = ["config_name", "query_text"]
    metadata_columns = ["query_type", "split", "method", "top_k"]
    for (config_name, query_text), group in per_query.groupby(group_columns, dropna=False):
        first = group.iloc[0]
        retrieved_item_ids = [str(item_id) for item_id in first["retrieved_item_ids"]]
        relevant_item_ids = {str(item_id) for item_id in group["target_item_id"].tolist()}
        row: dict[str, Any] = {
            "config_name": config_name,
            "query_text": query_text,
            "num_relevant_items": len(relevant_item_ids),
        }
        for column in metadata_columns:
            if column in group.columns:
                row[column] = first[column]
        for k in k_values:
            retrieved_at_k = {str(item_id) for item_id in retrieved_item_ids[:k]}
            hits = retrieved_at_k.intersection(relevant_item_ids)
            first_rank = _first_relevant_rank(retrieved_item_ids, relevant_item_ids, k)
            row[f"hit_any_at_{k}"] = 1.0 if hits else 0.0
            row[f"recall_multi_at_{k}"] = len(hits) / max(1, len(relevant_item_ids))
            row[f"mrr_any_at_{k}"] = 1.0 / first_rank if first_rank else 0.0
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["config_name", "query_text"]).reset_index(drop=True)


def summarize_multi_relevant_metrics(multi_relevant: pd.DataFrame) -> pd.DataFrame:
    """Summarize multi-relevant per-query-text metrics by config."""
    if multi_relevant.empty:
        return pd.DataFrame()

    metric_columns = [
        column
        for column in multi_relevant.columns
        if column.startswith("hit_any_at_")
        or column.startswith("recall_multi_at_")
        or column.startswith("mrr_any_at_")
    ]
    rows: list[dict[str, Any]] = []
    for config_name, group in multi_relevant.groupby("config_name", dropna=False):
        row: dict[str, Any] = {
            "config_name": config_name,
            "num_unique_query_texts": int(len(group)),
            "avg_relevant_items_per_query_text": float(group["num_relevant_items"].mean()),
            "max_relevant_items_per_query_text": int(group["num_relevant_items"].max()),
        }
        for column in sorted(metric_columns):
            row[column] = float(group[column].mean())
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


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
    rerank = config.get("rerank", {})
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
        "rerank": rerank,
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
            "rerank",
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
                "rerank": _format_config_value(config_summary["rerank"]),
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


def _evaluation_settings(config_path: Path) -> tuple[int, list[int]]:
    config = load_yaml_config(config_path)
    evaluation = config.get("evaluation", {})
    k_values = [int(k) for k in evaluation.get("k_values", [10, 20, 50])]
    top_k = int(config.get("retrieval", {}).get("top_k", max(k_values)))
    return top_k, k_values


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
    multi_relevant_summary: pd.DataFrame,
    artifact_paths: dict[str, str] | None = None,
) -> str:
    """Build a Markdown log for the retrieval tuning run."""
    summaries = [benchmark, *candidates]
    metadata = [
        _experiment_metadata(benchmark_config),
        *[_experiment_metadata(candidate_config) for candidate_config in candidate_configs],
    ]
    config_paths = [benchmark_config, *candidate_configs]
    change_rows = _build_change_rows(config_paths, metadata)
    benchmark_input_paths = require_input_paths(load_yaml_config(benchmark_config).get("input", {}))
    num_items = int(len(read_parquet(benchmark_input_paths["item_metadata_path"])))
    thresholds = _threshold_profile(num_items)
    threshold_text = _format_thresholds(thresholds)
    multi_relevant_lookup = {
        str(row["config_name"]): row
        for row in multi_relevant_summary.to_dict("records")
        if "config_name" in row
    }
    eligible_summaries = [
        (index, summary)
        for index, summary in enumerate(summaries)
        if _pool_pass_mode(summary, multi_relevant_lookup.get(metadata[index]["name"]), thresholds)
    ]
    if not eligible_summaries:
        eligible_summaries = list(enumerate(summaries))
    best_index, _ = max(
        eligible_summaries,
        key=lambda pair: (
            float(pair[1].get(RETRIEVAL_PASS_METRIC, 0.0)),
            float(
                multi_relevant_lookup.get(metadata[pair[0]]["name"], {}).get(
                    "recall_multi_at_50",
                    0.0,
                )
            ),
            float(
                multi_relevant_lookup.get(metadata[pair[0]]["name"], {}).get(
                    "hit_any_at_50",
                    0.0,
                )
            ),
            float(pair[1].get(RETRIEVAL_TIEBREAK_METRIC, 0.0)),
            -float(pair[1].get("avg_latency_ms", 0.0)),
        ),
    )
    selected_summary = summaries[best_index]
    selected_name = metadata[best_index]["name"]
    selected_multi_row = multi_relevant_lookup.get(selected_name)
    selected_pool_mode = _pool_pass_mode(selected_summary, selected_multi_row, thresholds)
    selected_passes_pool = bool(selected_pool_mode)
    pool_pass_count = sum(
        1
        for index, summary in enumerate(summaries)
        if _pool_pass_mode(summary, multi_relevant_lookup.get(metadata[index]["name"]), thresholds)
    )
    result_rows = []
    for step_index, (summary, meta) in enumerate(zip(summaries, metadata, strict=True)):
        multi_row = multi_relevant_lookup.get(meta["name"], {})
        result_rows.append(
            {
                "step": step_index,
                "name": meta["name"],
                "selection": "**BEST**" if step_index == best_index else "",
                "change": meta["change"],
                "threshold": threshold_text,
                "pass_status": _pass_status(summary, multi_row, thresholds),
                "recall_at_10": summary.get("recall_at_10", 0.0),
                "recall_at_50": summary.get("recall_at_50", 0.0),
                "recall_at_100": summary.get("recall_at_100", pd.NA),
                "hit_any_at_50": multi_row.get("hit_any_at_50", pd.NA),
                "recall_multi_at_50": multi_row.get("recall_multi_at_50", pd.NA),
                "mrr_at_10": summary.get("mrr_at_10", 0.0),
                "avg_latency_ms": summary.get("avg_latency_ms", 0.0),
            }
        )
    step_result_columns = [
        "step",
        "name",
        "selection",
        "change",
        "threshold",
        "pass_status",
        "recall_at_10",
        "recall_at_50",
        "hit_any_at_50",
        "recall_multi_at_50",
        "mrr_at_10",
        "avg_latency_ms",
    ]
    if any(not pd.isna(row.get("recall_at_100")) for row in result_rows):
        step_result_columns.insert(step_result_columns.index("mrr_at_10"), "recall_at_100")

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

    breakdown_rows = breakdown.to_dict("records")
    multi_relevant_rows = multi_relevant_summary.to_dict("records")
    multi_relevant_columns = [
        "config_name",
        "num_unique_query_texts",
        "avg_relevant_items_per_query_text",
        "max_relevant_items_per_query_text",
        "hit_any_at_50",
        "recall_multi_at_50",
    ]
    multi_relevant_columns = [
        column for column in multi_relevant_columns if column in multi_relevant_summary.columns
    ]
    interpretation_rows: list[dict[str, Any]] = []
    if not multi_relevant_summary.empty:
        benchmark_multi = multi_relevant_summary[
            multi_relevant_summary["config_name"] == metadata[0]["name"]
        ].iloc[0]
        interpretation_rows.append(
            {
                "question": "Does single-target scoring understate retrieval?",
                "observation": (
                    f"benchmark recall_at_50={float(benchmark.get('recall_at_50', 0.0)):.4f}; "
                    f"multi hit_any_at_50={float(benchmark_multi.get('hit_any_at_50', 0.0)):.4f}"
                ),
                "implication": (
                    "Compare these values to see whether single-target labels understate "
                    "candidate-pool coverage for broad queries."
                ),
            }
        )
        if "recall_multi_at_50" in multi_relevant_summary.columns:
            selected_multi = multi_relevant_summary[
                multi_relevant_summary["config_name"] == selected_name
            ].iloc[0]
            best_multi_value = float(multi_relevant_summary["recall_multi_at_50"].max())
            selected_multi_value = float(selected_multi["recall_multi_at_50"])
            tied_best_count = int(
                (multi_relevant_summary["recall_multi_at_50"].astype(float) == best_multi_value)
                .sum()
            )
            if selected_multi_value == best_multi_value and tied_best_count > 1:
                multi_note = " and ties for best"
            elif selected_multi_value == best_multi_value:
                multi_note = " and is best"
            else:
                multi_note = ""
            interpretation_rows.append(
                {
                    "question": "Which config gives the strongest multi-relevant candidate pool?",
                    "observation": (
                        f"{selected_name} has recall_multi_at_50="
                        f"{selected_multi_value:.4f}{multi_note}"
                    ),
                    "implication": (
                        "Use this as supporting evidence for the selected config, but keep "
                        "the pass/fail decision tied to the retrieval threshold."
                    ),
                }
            )
    benchmark_per_query = per_query[per_query["config_name"] == metadata[0]["name"]]
    query_item_pairs = _load_query_pairs_for_config(benchmark_config)
    all_query_item_pairs = read_parquet(benchmark_input_paths["query_item_pairs_path"])
    split_counts = all_query_item_pairs["split"].value_counts()
    dataset_rows = [
        {
            "num_items": num_items,
            "num_query_pairs": int(len(all_query_item_pairs)),
            "train_query_pairs": int(split_counts.get("train", 0)),
            "val_query_pairs": int(split_counts.get("val", 0)),
            "test_query_pairs": int(split_counts.get("test", 0)),
            "num_eval_rows": int(len(benchmark_per_query)),
            "unique_query_texts": int(query_item_pairs["query_text"].nunique()),
            "repeated_query_text_rate": 1.0
            - float(query_item_pairs["query_text"].nunique() / max(1, len(query_item_pairs))),
            "strict_top_k": STRICT_TOP_K,
        }
    ]
    best_recall_10 = max(summaries, key=lambda row: float(row.get("recall_at_10", 0.0)))
    best_mrr_10 = max(summaries, key=lambda row: float(row.get("mrr_at_10", 0.0)))
    if selected_pool_mode == "single":
        candidate_pool_takeaway = (
            f"Top-50 candidate generation passes the single-target pool gate for this "
            f"{num_items}-item retrieval set."
        )
    elif selected_pool_mode == "multi":
        candidate_pool_takeaway = (
            f"Top-50 candidate generation passes the multi-relevant pool gate for this "
            f"{num_items}-item retrieval set, despite missing the single-target Recall@50 gate."
        )
    else:
        candidate_pool_takeaway = (
            f"Top-50 candidate generation is below the single and multi pool gates for this "
            f"{num_items}-item retrieval set."
        )
    if pool_pass_count == len(summaries):
        next_evidence = "All retained BM25 retrieval variants pass the retrieval pool gate."
        next_takeaway = (
            "Choose the candidate generator by pool coverage first, then Recall@10 and latency "
            "as tie-breaks."
        )
    elif pool_pass_count > 0:
        next_evidence = f"{pool_pass_count} retained BM25 variant(s) pass the retrieval pool gate."
        next_takeaway = (
            "Use the selected passing config as a tentative retrieval final, then validate "
            "ranking with that candidate source."
        )
    else:
        next_evidence = "No retained BM25 variant passes the retrieval pool gate."
        next_takeaway = (
            "Do not mark retrieval final yet; tune candidate generation or revisit the "
            "evaluation labels before ranking tuning."
        )
    decision_rows = [
        {
            "decision_point": "selected_config",
            "evidence": (
                f"{selected_name} selected by Recall@50, Recall@10, then latency."
            ),
            "retrieval_takeaway": (
                "Use this as the retrieval candidate generator for the next stage."
                if selected_passes_pool
                else "Treat this as the best observed retrieval config, not yet a final handoff."
            ),
        },
        {
            "decision_point": "candidate_pool",
            "evidence": (
                f"selected Recall@50={float(selected_summary.get('recall_at_50', 0.0)):.4f} "
                f"({selected_name})"
            ),
            "retrieval_takeaway": candidate_pool_takeaway,
        },
        {
            "decision_point": "top10_recall",
            "evidence": (
                f"best Recall@10={float(best_recall_10.get('recall_at_10', 0.0)):.4f} "
                f"({metadata[summaries.index(best_recall_10)]['name']})"
            ),
            "retrieval_takeaway": (
                "Use Recall@10 as a tie-break among configs that pass Recall@50; it is not "
                "the retrieval pass/fail criterion."
            ),
        },
        {
            "decision_point": "early_rank",
            "evidence": (
                f"best MRR@10={float(best_mrr_10.get('mrr_at_10', 0.0)):.4f} "
                f"({metadata[summaries.index(best_mrr_10)]['name']})"
            ),
            "retrieval_takeaway": (
                "Use MRR@10 only as an ordering diagnostic; candidate generation is judged "
                "by Recall@50."
            ),
        },
        {
            "decision_point": "next_action",
            "evidence": next_evidence,
            "retrieval_takeaway": next_takeaway,
        },
    ]

    artifacts = artifact_paths or {
        "summary_results": "validation/results/retrieval_tuning_results.csv",
        "comparison": "validation/results/retrieval_tuning_comparison.csv",
        "strict_comparison": "validation/results/retrieval_tuning_strict_comparison.csv",
        "per_query": "validation/results/retrieval_tuning_per_query.csv",
        "query_type_breakdown": "validation/results/retrieval_tuning_query_type_breakdown.csv",
        "multi_relevant": "validation/results/retrieval_tuning_multi_relevant.csv",
        "log": "validation/reports/retrieval_tuning_log.md",
    }
    artifact_lines = "\n".join(f"- `{path}`" for path in artifacts.values())

    return (
        "\n\n".join(
            [
                "# Retrieval Tuning Log",
                "This log compares each retrieval tuning step against the fixed BM25 benchmark "
                "using the configured synthetic data.",
                "## Dataset Notes",
                _markdown_table(
                    dataset_rows,
                    [
                        "num_items",
                        "num_query_pairs",
                        "train_query_pairs",
                        "val_query_pairs",
                        "test_query_pairs",
                        "num_eval_rows",
                        "unique_query_texts",
                        "repeated_query_text_rate",
                        "strict_top_k",
                    ],
                ),
                "Retrieval pass/fail is based on `Recall@50`, because this stage is evaluated "
                "as a candidate generator. `Recall@10` is a tie-break signal, and `MRR@10` is "
                "kept only as a diagnostic for early ordering.",
                "## Evaluation Change Log",
                "Added multi-relevant retrieval evaluation. The original single-target metrics "
                "score each query-item row against exactly one target item. The multi-relevant "
                "view groups rows by `query_text`, treats every target item for that text as "
                "relevant, and reports candidate-pool coverage at 50. This does not change the "
                "retriever; it only clarifies whether broad synthetic queries are being judged "
                "too narrowly.",
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
                _markdown_table(result_rows, step_result_columns),
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
                "## Multi-Relevant Evaluation",
                _markdown_table(multi_relevant_rows, multi_relevant_columns),
                "`hit_any_at_50` asks whether at least one relevant item for the query text was "
                "retrieved in the candidate pool. `recall_multi_at_50` asks what fraction of "
                "all known relevant items for that query text were retrieved.",
                "## Evaluation Interpretation",
                _markdown_table(
                    interpretation_rows,
                    ["question", "observation", "implication"],
                ),
                "## Retrieval Decision",
                _markdown_table(
                    decision_rows,
                    ["decision_point", "evidence", "retrieval_takeaway"],
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
                "## Output Artifacts",
                artifact_lines,
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
    multi_relevant_path: str | Path = DEFAULT_MULTI_RELEVANT_PATH,
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
    full_top_k, full_k_values = _evaluation_settings(benchmark_path)
    full_per_query_frames = [
        _evaluate_config_per_query(path, meta["name"], top_k=full_top_k, k_values=full_k_values)
        for path, meta in zip(all_paths, all_metadata, strict=True)
    ]
    full_per_query = pd.concat(full_per_query_frames, ignore_index=True)
    multi_relevant = build_multi_relevant_metrics(full_per_query, full_k_values)
    multi_relevant_summary = summarize_multi_relevant_metrics(multi_relevant)
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

    resolved_multi_relevant_path = _resolve(multi_relevant_path)
    resolved_multi_relevant_path.parent.mkdir(parents=True, exist_ok=True)
    multi_relevant.to_csv(resolved_multi_relevant_path, index=False)
    logger.info(
        "Retrieval tuning multi-relevant output saved to %s",
        resolved_multi_relevant_path,
    )

    resolved_log_path = _resolve(log_path)
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_output_config = load_yaml_config(benchmark_path).get("output", {})
    summary_results_path = _resolve(benchmark_output_config["results_path"])
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
            multi_relevant_summary,
            artifact_paths={
                "summary_results": str(summary_results_path),
                "comparison": str(resolved_output_path),
                "strict_comparison": str(resolved_strict_path),
                "per_query": str(resolved_per_query_path),
                "query_type_breakdown": str(resolved_breakdown_path),
                "multi_relevant": str(resolved_multi_relevant_path),
                "log": str(resolved_log_path),
            },
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
    parser.add_argument(
        "--per-query-output",
        type=Path,
        default=DEFAULT_PER_QUERY_PATH,
        help="Per-query strict retrieval output path.",
    )
    parser.add_argument(
        "--breakdown-output",
        type=Path,
        default=DEFAULT_BREAKDOWN_PATH,
        help="Query-type breakdown output path.",
    )
    parser.add_argument(
        "--strict-output",
        type=Path,
        default=DEFAULT_STRICT_PATH,
        help="Strict top-k comparison output path.",
    )
    parser.add_argument(
        "--multi-relevant-output",
        type=Path,
        default=DEFAULT_MULTI_RELEVANT_PATH,
        help="Multi-relevant retrieval output path.",
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
        per_query_path=args.per_query_output,
        breakdown_path=args.breakdown_output,
        strict_path=args.strict_output,
        multi_relevant_path=args.multi_relevant_output,
    )


if __name__ == "__main__":
    main()
