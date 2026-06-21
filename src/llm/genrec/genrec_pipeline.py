"""Pipeline orchestration for Stage 10 GenRec experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.genrec_eval import (
    evaluate_explanations,
    evaluate_genrec_outputs,
    summarize_genrec_metrics,
)
from src.llm.clients.base_client import BaseLLMClient
from src.llm.explanations.explanation_generator import (
    LLMExplanationGenerator,
    TemplateExplanationGenerator,
)
from src.llm.genrec.candidate_formatter import (
    format_candidates_for_llm,
    load_candidate_source,
    select_top_candidates_for_query,
)
from src.llm.genrec.constrained_generator import CandidateConstrainedGenerator
from src.llm.genrec.direct_generator import DirectGenerator
from src.llm.genrec.llm_reranker import LLMReranker
from src.llm.genrec.output_parser import fallback_recommendations_from_candidates
from src.llm.query_understanding.query_understanding_pipeline import build_llm_client_from_config
from src.utils.config import resolve_project_path
from src.utils.io import read_parquet, write_parquet

GENREC_OUTPUT_COLUMNS = [
    "query_id",
    "query_text",
    "split",
    "target_item_id",
    "method",
    "candidate_pool_size",
    "recommended_item_ids",
    "ranked_items",
    "invalid_item_ids",
    "used_fallback",
    "parse_success",
    "schema_valid",
    "parse_error",
    "provider",
    "model",
    "latency_ms",
    "estimated_cost_usd",
]

EXPLANATION_OUTPUT_COLUMNS = [
    "query_id",
    "query_text",
    "split",
    "target_item_id",
    "method",
    "recommended_item_ids",
    "explanations",
    "parse_success",
    "schema_valid",
    "used_fallback",
    "parse_error",
    "explanation_faithful",
    "ungrounded_claim_count",
    "missing_evidence_count",
    "provider",
    "model",
    "latency_ms",
    "estimated_cost_usd",
]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolve_project_path(str(candidate))


def build_genrec_client_from_config(config: dict[str, Any]) -> BaseLLMClient:
    """Build an LLM client using the existing Stage 8 abstraction."""
    return build_llm_client_from_config(config)


def _read_optional_parquet(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    resolved = _resolve_path(path)
    return read_parquet(resolved) if resolved.exists() else pd.DataFrame()


def load_genrec_inputs(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Load Stage 10 input artifacts."""
    input_config = config.get("input", {})
    query_pairs = read_parquet(_resolve_path(input_config["query_item_pairs_path"]))
    items = read_parquet(_resolve_path(input_config["item_metadata_path"]))
    ranking_candidates = load_candidate_source(config)
    return {
        "query_item_pairs": query_pairs,
        "items": items,
        "ranking_candidates": ranking_candidates,
        "parsed_queries": _read_optional_parquet(input_config.get("llm_query_understanding_path")),
        "user_profiles": _read_optional_parquet(input_config.get("user_profiles_path")),
    }


def _parsed_query_lookup(parsed_queries: pd.DataFrame) -> dict[str, dict[str, Any]]:
    lookup = {}
    if parsed_queries.empty or "query_id" not in parsed_queries.columns:
        return lookup
    for _, row in parsed_queries.iterrows():
        lookup[str(row["query_id"])] = {
            "intent": row.get("llm_intent"),
            "category": row.get("llm_category"),
            "brand": row.get("llm_brand"),
            "price_constraint": row.get("llm_price_constraint"),
            "use_case": row.get("llm_use_case"),
            "rewritten_query": row.get("rewritten_query"),
            "expanded_queries": row.get("expanded_queries", []),
        }
    return lookup


def _user_profile_lookup(user_profiles: pd.DataFrame) -> dict[str, dict[str, Any]]:
    lookup = {}
    if user_profiles.empty or "user_id" not in user_profiles.columns:
        return lookup
    for _, row in user_profiles.iterrows():
        lookup[str(row["user_id"])] = {
            "user_id": row.get("user_id"),
            "summary": row.get("summary"),
            "top_categories": row.get("top_categories", []),
            "top_brands": row.get("top_brands", []),
            "price_preference": row.get("price_preference"),
            "profile_text": row.get("profile_text"),
        }
    return lookup


def build_query_contexts(
    query_item_pairs: pd.DataFrame,
    items: pd.DataFrame,
    ranking_candidates: pd.DataFrame,
    parsed_queries: pd.DataFrame | None = None,
    user_profiles: pd.DataFrame | None = None,
    split: str = "test",
    candidate_pool_size: int = 20,
    max_queries: int | None = None,
) -> list[dict[str, Any]]:
    """Build deterministic query contexts for Stage 10 experiments."""
    pairs = query_item_pairs.copy()
    if split:
        pairs = pairs[pairs["split"].astype(str) == str(split)].copy()
    pairs = pairs.sort_values(["split", "query_id"]).reset_index(drop=True)
    if max_queries is not None:
        pairs = pairs.head(int(max_queries)).copy()

    parsed_lookup = _parsed_query_lookup(
        parsed_queries if parsed_queries is not None else pd.DataFrame()
    )
    profile_lookup = _user_profile_lookup(
        user_profiles if user_profiles is not None else pd.DataFrame()
    )
    contexts = []
    for _, row in pairs.iterrows():
        query_id = str(row["query_id"])
        top_rows = select_top_candidates_for_query(
            ranking_candidates,
            query_id=query_id,
            top_n=candidate_pool_size,
        )
        candidates = format_candidates_for_llm(top_rows, items, max_candidates=candidate_pool_size)
        user_profile = None
        if "user_id" in row and pd.notna(row["user_id"]):
            user_profile = profile_lookup.get(str(row["user_id"]))
        contexts.append(
            {
                "query_id": query_id,
                "query_text": str(row.get("query_text", "")),
                "split": str(row.get("split", "")),
                "target_item_id": str(row.get("target_item_id", "")),
                "parsed_query": parsed_lookup.get(query_id),
                "user_profile": user_profile,
                "candidates": candidates,
                "candidate_rows": top_rows,
            }
        )
    return contexts


def run_direct_generation_experiment(
    client: BaseLLMClient,
    contexts: list[dict[str, Any]],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Run the direct-generation baseline."""
    llm_config = config.get("llm", {})
    generator = DirectGenerator(
        client,
        temperature=float(llm_config.get("temperature", 0.0)),
        max_tokens=int(llm_config.get("max_tokens", 1024)),
    )
    return generator.batch_generate(
        contexts,
        top_k=int(config.get("genrec", {}).get("final_top_k", 10)),
    )


def run_candidate_constrained_experiment(
    client: BaseLLMClient,
    contexts: list[dict[str, Any]],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Run candidate-constrained generation."""
    llm_config = config.get("llm", {})
    generator = CandidateConstrainedGenerator(
        client,
        temperature=float(llm_config.get("temperature", 0.0)),
        max_tokens=int(llm_config.get("max_tokens", 1024)),
        allow_fallback=bool(
            config.get("constraints", {}).get("allow_fallback_to_ranked_candidates", True)
        ),
    )
    return generator.batch_generate(
        contexts,
        top_k=int(config.get("genrec", {}).get("final_top_k", 10)),
    )


def run_llm_rerank_experiment(
    client: BaseLLMClient,
    contexts: list[dict[str, Any]],
    config: dict[str, Any],
    candidate_pool_size: int,
) -> pd.DataFrame:
    """Run LLM reranking for a configured candidate pool size."""
    llm_config = config.get("llm", {})
    reranker = LLMReranker(
        client,
        candidate_pool_size=candidate_pool_size,
        temperature=float(llm_config.get("temperature", 0.0)),
        max_tokens=int(llm_config.get("max_tokens", 1024)),
        allow_fallback=bool(
            config.get("constraints", {}).get("allow_fallback_to_ranked_candidates", True)
        ),
    )
    return reranker.batch_rerank(
        contexts,
        top_k=int(config.get("genrec", {}).get("final_top_k", 10)),
    )


def _base_recommendations_for_explanations(
    contexts: list[dict[str, Any]],
    constrained_outputs: pd.DataFrame,
    top_k: int,
) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    if not constrained_outputs.empty:
        for _, row in constrained_outputs.iterrows():
            lookup[str(row["query_id"])] = list(row.get("recommended_item_ids", []))
    for context in contexts:
        query_id = str(context["query_id"])
        if query_id not in lookup:
            fallback = fallback_recommendations_from_candidates(
                context.get("candidates", []),
                top_k=top_k,
            )
            lookup[query_id] = list(fallback["recommended_item_ids"])
    return lookup


def run_explanation_experiment(
    client: BaseLLMClient,
    contexts: list[dict[str, Any]],
    items: pd.DataFrame,
    config: dict[str, Any],
    method: str,
    base_recommendations: dict[str, list[str]],
) -> pd.DataFrame:
    """Run template or evidence-grounded LLM explanations."""
    explanation_config = config.get("explanations", {})
    max_evidence_items = int(explanation_config.get("max_evidence_items", 5))
    if method == "template_explanation":
        generator: Any = TemplateExplanationGenerator()
    else:
        llm_config = config.get("llm", {})
        generator = LLMExplanationGenerator(
            client,
            temperature=float(llm_config.get("temperature", 0.0)),
            max_tokens=int(llm_config.get("max_tokens", 1024)),
        )
    rows = []
    for context in contexts:
        query_id = str(context["query_id"])
        recommended_ids = base_recommendations.get(query_id, [])
        output = generator.generate(
            query_id=query_id,
            query_text=str(context.get("query_text", "")),
            recommended_item_ids=recommended_ids,
            items=items,
            ranking_candidates_for_query=context.get("candidate_rows"),
            parsed_query=context.get("parsed_query"),
            user_profile=context.get("user_profile"),
            max_evidence_items=max_evidence_items,
        )
        output.update(
            {
                "query_id": query_id,
                "query_text": str(context.get("query_text", "")),
                "split": str(context.get("split", "")),
                "target_item_id": str(context.get("target_item_id", "")),
                "recommended_item_ids": recommended_ids,
            }
        )
        rows.append(output)
    return pd.DataFrame(rows)


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = df.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = None
    return output[columns]


def run_genrec_pipeline(
    config: dict[str, Any],
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run configured Stage 10 experiments and save parquet artifacts."""
    inputs = load_genrec_inputs(config)
    genrec_config = config.get("genrec", {})
    split = str(genrec_config.get("split", "test"))
    max_queries = genrec_config.get("max_queries")
    contexts = build_query_contexts(
        query_item_pairs=inputs["query_item_pairs"],
        items=inputs["items"],
        ranking_candidates=inputs["ranking_candidates"],
        parsed_queries=inputs["parsed_queries"],
        user_profiles=inputs["user_profiles"],
        split=split,
        candidate_pool_size=int(genrec_config.get("candidate_pool_size", 20)),
        max_queries=int(max_queries) if max_queries is not None else None,
    )
    client = build_genrec_client_from_config(config)
    experiments = set(genrec_config.get("experiments", []))
    top_k = int(genrec_config.get("final_top_k", 10))
    genrec_frames = []
    rerank_frames = []
    explanation_frames = []
    constrained_outputs = pd.DataFrame()

    if "direct_generation" in experiments:
        genrec_frames.append(run_direct_generation_experiment(client, contexts, config))
    if "candidate_constrained_generation" in experiments:
        constrained_outputs = run_candidate_constrained_experiment(client, contexts, config)
        genrec_frames.append(constrained_outputs)
    for experiment in sorted(experiments):
        if experiment.startswith("llm_rerank_top_"):
            pool_size = int(experiment.rsplit("_", 1)[-1])
            rerank_frames.append(run_llm_rerank_experiment(client, contexts, config, pool_size))
    if {"template_explanation", "evidence_grounded_llm_explanation"} & experiments:
        if constrained_outputs.empty:
            constrained_outputs = run_candidate_constrained_experiment(client, contexts, config)
        base_recommendations = _base_recommendations_for_explanations(
            contexts,
            constrained_outputs,
            top_k=top_k,
        )
        for method in ["template_explanation", "evidence_grounded_llm_explanation"]:
            if method in experiments:
                explanation_frames.append(
                    run_explanation_experiment(
                        client,
                        contexts,
                        inputs["items"],
                        config,
                        method,
                        base_recommendations,
                    )
                )

    genrec_outputs = (
        pd.concat(genrec_frames, ignore_index=True)
        if genrec_frames
        else pd.DataFrame(columns=GENREC_OUTPUT_COLUMNS)
    )
    rerank_outputs = (
        pd.concat(rerank_frames, ignore_index=True)
        if rerank_frames
        else pd.DataFrame(columns=GENREC_OUTPUT_COLUMNS)
    )
    explanations = (
        pd.concat(explanation_frames, ignore_index=True)
        if explanation_frames
        else pd.DataFrame(columns=EXPLANATION_OUTPUT_COLUMNS)
    )
    genrec_outputs = _ensure_columns(genrec_outputs, GENREC_OUTPUT_COLUMNS)
    rerank_outputs = _ensure_columns(rerank_outputs, GENREC_OUTPUT_COLUMNS)
    explanations = _ensure_columns(explanations, EXPLANATION_OUTPUT_COLUMNS)

    output_config = config.get("output", {})
    genrec_path = _resolve_path(output_config["genrec_outputs_path"])
    rerank_path = _resolve_path(output_config["llm_rerank_outputs_path"])
    explanations_path = _resolve_path(output_config["explanations_path"])
    write_parquet(genrec_outputs, genrec_path)
    write_parquet(rerank_outputs, rerank_path)
    write_parquet(explanations, explanations_path)

    scored_frames = [frame for frame in [genrec_outputs, rerank_outputs] if not frame.empty]
    scored_outputs = (
        pd.concat(scored_frames, ignore_index=True)
        if scored_frames
        else pd.DataFrame(columns=GENREC_OUTPUT_COLUMNS)
    )
    per_query = evaluate_genrec_outputs(
        scored_outputs,
        query_item_pairs=inputs["query_item_pairs"],
        item_metadata=inputs["items"],
        k_values=[int(k) for k in config.get("evaluation", {}).get("k_values", [10])],
    )
    explanation_eval = evaluate_explanations(explanations)
    metric_rows = pd.concat([per_query, explanation_eval], ignore_index=True)
    summary = summarize_genrec_metrics(metric_rows)
    if not summary.empty:
        summary["config_path"] = str(config_path or "")

    return {
        "genrec_outputs": genrec_outputs,
        "rerank_outputs": rerank_outputs,
        "explanations": explanations,
        "per_query_metrics": metric_rows,
        "summary": summary,
        "genrec_outputs_path": str(genrec_path),
        "llm_rerank_outputs_path": str(rerank_path),
        "explanations_path": str(explanations_path),
        "num_queries": len(contexts),
        "methods": sorted(experiments),
    }
