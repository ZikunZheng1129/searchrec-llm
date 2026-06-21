"""Evaluate Stage 10 GenRec, LLM reranking, and explanations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.report import render_markdown_table  # noqa: E402
from src.llm.genrec.genrec_pipeline import run_genrec_pipeline  # noqa: E402
from src.utils.config import load_yaml_config, resolve_project_path  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

RESULT_COLUMNS = [
    "stage",
    "method",
    "provider",
    "model",
    "split",
    "num_queries",
    "valid_item_rate",
    "hallucination_rate",
    "output_parse_success_rate",
    "schema_valid_rate",
    "fallback_rate",
    "ndcg_at_10",
    "mrr_at_10",
    "recall_at_10",
    "explanation_faithfulness",
    "avg_latency_ms",
    "p95_latency_ms",
    "estimated_cost_per_1000_queries",
    "config_path",
]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolve_project_path(str(candidate))


def _write_results(summary: pd.DataFrame, results_path: Path, config_path: str | Path) -> None:
    output = summary.copy()
    output["config_path"] = str(config_path)
    for column in RESULT_COLUMNS:
        if column not in output.columns:
            text_columns = {"stage", "method", "provider", "model", "split", "config_path"}
            output[column] = "" if column in text_columns else np.nan
    output = output[RESULT_COLUMNS]
    if results_path.exists():
        existing = pd.read_csv(results_path)
        if set(RESULT_COLUMNS).issubset(existing.columns):
            keys = ["method", "provider", "model", "split", "config_path"]
            for _, row in output.iterrows():
                same_run = pd.Series(True, index=existing.index)
                for key in keys:
                    same_run &= existing[key].astype(str) == str(row[key])
                existing = existing.loc[~same_run, RESULT_COLUMNS]
        else:
            existing = pd.DataFrame(columns=RESULT_COLUMNS)
        output = pd.concat([existing, output], ignore_index=True)
    output = output.sort_values(["method", "provider", "model", "split", "config_path"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(results_path, index=False)


def _best_candidate_constrained(results: pd.DataFrame) -> str:
    preferred = results[
        results["method"]
        .astype(str)
        .isin(
            [
                "candidate_constrained_generation",
                "llm_rerank_top_10",
                "llm_rerank_top_20",
                "evidence_grounded_llm_explanation",
            ]
        )
    ].copy()
    if preferred.empty:
        return "No constrained GenRec method has been evaluated yet."
    preferred = preferred.sort_values(
        ["valid_item_rate", "output_parse_success_rate", "ndcg_at_10", "avg_latency_ms", "method"],
        ascending=[False, False, False, True, True],
    )
    row = preferred.iloc[0]
    return (
        f"`{row['method']}` is the current local choice among constrained methods: "
        f"valid_item_rate={row['valid_item_rate']:.4f}, "
        f"hallucination_rate={row['hallucination_rate']:.4f}, "
        f"parse_success={row['output_parse_success_rate']:.4f}, "
        f"ndcg_at_10={row['ndcg_at_10']:.4f}."
    )


def generate_final_llm_decision_report(
    genrec_results: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Generate the Stage 10 LLM decision report from real local results."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    display_columns = [
        "method",
        "provider",
        "model",
        "num_queries",
        "valid_item_rate",
        "hallucination_rate",
        "output_parse_success_rate",
        "fallback_rate",
        "ndcg_at_10",
        "mrr_at_10",
        "explanation_faithfulness",
        "avg_latency_ms",
    ]
    table = render_markdown_table(genrec_results[display_columns])
    direct_rows = genrec_results[genrec_results["method"] == "direct_generation"]
    direct_risk = "Direct generation has not been evaluated yet."
    if not direct_rows.empty:
        row = direct_rows.iloc[0]
        direct_risk = (
            f"Direct generation valid_item_rate={row['valid_item_rate']:.4f} and "
            f"hallucination_rate={row['hallucination_rate']:.4f}. It is retained only "
            "as a baseline for invalid item risk."
        )

    text = f"""# Final LLM Decision Report

Stage 10 status: local mock GenRec, candidate-constrained reranking/generation,
and evidence-grounded explanations are implemented.

Data caveat: current measurements use synthetic debug data and the deterministic
mock LLM client by default. These metrics validate mechanics and safety checks,
not production LLM quality.

## LLM Components Evaluated

- Stage 8 query understanding.
- Stage 8 user profile generation when user context exists.
- Stage 10 direct generation baseline.
- Stage 10 candidate-constrained generation.
- Stage 10 LLM reranking.
- Stage 10 template and evidence-grounded explanations.

## Metrics

{table}

## Direct Generation Risk

{direct_risk}

Free-form direct generation is not catalog-grounded. It can emit invalid item IDs,
which creates hallucination and serving risk.

## Candidate-Constrained Design

- Selects only from retrieved/ranked catalog candidates.
- Validates every item ID before accepting output.
- Falls back to original ranked candidates if parsing or validation fails.
- Generates explanations from item metadata and candidate evidence only.

## Final Local Decision

{_best_candidate_constrained(genrec_results)}

Use LLMs for query understanding, profile summarization, candidate-constrained
reranking/generation, and faithful explanation. Do not use free-form direct
generation as the final recommendation output.

## Not Evaluated Yet

- Real LLM quality.
- Production traffic.
- Human preference evaluation.
- API/dashboard demo.

## Interview Talking Point

“We tested direct generation and candidate-constrained GenRec. Direct generation
had invalid item risk, so the final design grounds LLM outputs in retrieved
catalog candidates and validates every item ID before serving.”
"""
    output.write_text(text, encoding="utf-8")


def run_evaluate_genrec(config_path: str | Path) -> dict[str, Any]:
    """Run Stage 10 experiments, update metrics, and write decision report."""
    logger = get_logger("tiksearchrec.evaluate_genrec")
    config = load_yaml_config(config_path)
    set_seed(int(config.get("seed", 42)))
    result = run_genrec_pipeline(config, config_path=config_path)
    summary = result["summary"]
    results_path = _resolve_path(config.get("output", {})["results_path"])
    if summary.empty:
        raise ValueError("GenRec pipeline produced no summary rows")
    _write_results(summary, results_path, config_path)
    all_results = pd.read_csv(results_path)
    report_path = _resolve_path(config.get("output", {})["final_llm_decision_report_path"])
    generate_final_llm_decision_report(all_results, report_path)

    logger.info(
        "Evaluated GenRec provider=%s model=%s split=%s queries=%s methods=%s",
        summary["provider"].iloc[0],
        summary["model"].iloc[0],
        summary["split"].iloc[0],
        result["num_queries"],
        ", ".join(result["methods"]),
    )
    for _, row in summary.iterrows():
        logger.info(
            "%s valid=%s hallucination=%s parse=%s ndcg@10=%s mrr@10=%s fallback=%s latency=%s",
            row["method"],
            row["valid_item_rate"],
            row["hallucination_rate"],
            row["output_parse_success_rate"],
            row["ndcg_at_10"],
            row["mrr_at_10"],
            row["fallback_rate"],
            row["avg_latency_ms"],
        )
    logger.info("GenRec outputs saved to %s", result["genrec_outputs_path"])
    logger.info("LLM rerank outputs saved to %s", result["llm_rerank_outputs_path"])
    logger.info("Explanations saved to %s", result["explanations_path"])
    logger.info("Results saved to %s", results_path)
    logger.info("LLM decision report saved to %s", report_path)

    return {
        **result,
        "results_path": str(results_path),
        "final_llm_decision_report_path": str(report_path),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Stage 10 GenRec.")
    parser.add_argument("--config", type=Path, required=True, help="Path to GenRec config.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_evaluate_genrec(args.config)


if __name__ == "__main__":
    main()
