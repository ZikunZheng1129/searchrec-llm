"""Local artifact-backed demo service for the FastAPI and dashboard layers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from app.api.errors import ArtifactMissingError, artifact_missing_message
from src.llm.clients.mock_client import MockLLMClient
from src.llm.explanations.explanation_generator import TemplateExplanationGenerator
from src.llm.genrec.candidate_formatter import (
    DEFAULT_SCORE_PRIORITY,
    format_candidates_for_llm,
    select_top_candidates_for_query,
)
from src.llm.genrec.constrained_generator import CandidateConstrainedGenerator
from src.llm.genrec.llm_reranker import LLMReranker
from src.llm.query_understanding.intent_extractor import LLMIntentExtractor
from src.retrieval.text_utils import build_item_text, tokenize
from src.utils.config import resolve_project_path

COMMAND_HINTS = {
    "item_metadata_path": (
        "python src/pipelines/build_dataset.py --config configs/data/debug_sample.yaml"
    ),
    "query_item_pairs_path": (
        "python src/pipelines/generate_queries.py --config configs/data/query_generation_debug.yaml"
    ),
    "ranking_candidates_path": (
        "python src/pipelines/build_ranking_dataset.py --config "
        "configs/ranking/ranking_dataset_debug.yaml"
    ),
    "ranking_candidates_multimodal_path": (
        "python src/pipelines/augment_ranking_with_multimodal.py --config "
        "configs/multimodal/multimodal_fusion_debug.yaml"
    ),
    "llm_query_understanding_path": (
        "python src/pipelines/evaluate_llm_query_understanding.py --config "
        "configs/llm/query_understanding.yaml"
    ),
    "user_profiles_path": (
        "python src/pipelines/generate_user_profiles.py --config configs/llm/user_profile.yaml"
    ),
    "genrec_outputs_path": (
        "python src/pipelines/evaluate_genrec.py --config configs/genrec/genrec_debug.yaml"
    ),
    "final_leaderboard_path": (
        "python src/pipelines/generate_validation_report.py --config "
        "validation/experiments/stage5_debug_validation.yaml"
    ),
}


class DemoService:
    """Small local demo facade over existing artifacts and mock LLM components."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.artifacts = config.get("artifacts", {})
        self.demo = config.get("demo", {})
        self._cache: dict[str, pd.DataFrame | str] = {}

    def _resolve(self, key: str) -> Path:
        raw = self.artifacts.get(key)
        if raw is None:
            raise KeyError(f"Unknown artifact key: {key}")
        path = Path(raw)
        return path if path.is_absolute() else resolve_project_path(str(path))

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): DemoService._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [DemoService._json_safe(item) for item in value]
        if hasattr(value, "tolist"):
            return DemoService._json_safe(value.tolist())
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return value

    @classmethod
    def _records(cls, df: pd.DataFrame) -> list[dict[str, Any]]:
        return [cls._json_safe(row) for row in df.to_dict(orient="records")]

    def _read_parquet(self, key: str, required: bool = True) -> pd.DataFrame:
        if key in self._cache and isinstance(self._cache[key], pd.DataFrame):
            return self._cache[key]  # type: ignore[return-value]
        if key not in self.artifacts and not required:
            return pd.DataFrame()
        path = self._resolve(key)
        if not path.exists():
            if required:
                raise ArtifactMissingError(path, COMMAND_HINTS.get(key))
            return pd.DataFrame()
        df = pd.read_parquet(path)
        self._cache[key] = df
        return df

    def _read_csv(self, key: str, required: bool = True) -> pd.DataFrame:
        if key in self._cache and isinstance(self._cache[key], pd.DataFrame):
            return self._cache[key]  # type: ignore[return-value]
        if key not in self.artifacts and not required:
            return pd.DataFrame()
        path = self._resolve(key)
        if not path.exists():
            if required:
                raise ArtifactMissingError(path, COMMAND_HINTS.get(key))
            return pd.DataFrame()
        df = pd.read_csv(path)
        self._cache[key] = df
        return df

    def _read_text(self, key: str, required: bool = False) -> str:
        if key in self._cache and isinstance(self._cache[key], str):
            return self._cache[key]  # type: ignore[return-value]
        if key not in self.artifacts and not required:
            return ""
        path = self._resolve(key)
        if not path.exists():
            if required:
                raise ArtifactMissingError(path, COMMAND_HINTS.get(key))
            return ""
        text = path.read_text(encoding="utf-8")
        self._cache[key] = text
        return text

    def artifact_status(self) -> dict[str, Any]:
        """Return existence status for configured artifacts."""
        required_keys = {
            "item_metadata_path",
            "query_item_pairs_path",
            "ranking_candidates_path",
            "final_leaderboard_path",
        }
        rows = []
        for key, raw_path in sorted(self.artifacts.items()):
            path = self._resolve(key)
            rows.append(
                {
                    "name": key,
                    "path": str(raw_path),
                    "exists": path.exists(),
                    "required": key in required_keys,
                    "command_hint": COMMAND_HINTS.get(key),
                }
            )
        return {"artifacts": rows}

    def health(self) -> dict[str, Any]:
        """Return local service health and artifact summary."""
        statuses = self.artifact_status()["artifacts"]
        available = sum(1 for row in statuses if row["exists"])
        app_config = self.config.get("app", {})
        return {
            "name": app_config.get("name", "TikSearchRec-LLM API"),
            "version": app_config.get("version", "0.11.0"),
            "environment": app_config.get("environment", "local"),
            "status": "ok" if available else "missing_artifacts",
            "artifacts_available": available,
            "artifacts_total": len(statuses),
        }

    def _items(self) -> pd.DataFrame:
        return self._read_parquet("item_metadata_path")

    def _query_pairs(self) -> pd.DataFrame:
        return self._read_parquet("query_item_pairs_path")

    def _candidate_key(self, use_multimodal: bool = True) -> str:
        if use_multimodal:
            try:
                if self._resolve("ranking_candidates_multimodal_path").exists():
                    return "ranking_candidates_multimodal_path"
            except KeyError:
                pass
        return "ranking_candidates_path"

    def _candidates(self, use_multimodal: bool = True) -> pd.DataFrame:
        return self._read_parquet(self._candidate_key(use_multimodal))

    def _item_lookup(self) -> dict[str, dict[str, Any]]:
        items = self._items()
        return {str(row["item_id"]): row.to_dict() for _, row in items.iterrows()}

    @staticmethod
    def _score_column(df: pd.DataFrame) -> str | None:
        for column in DEFAULT_SCORE_PRIORITY:
            if column in df.columns:
                return column
        return None

    def _enrich_item(
        self,
        item_id: str,
        score: float = 0.0,
        rank: int = 0,
        source: str = "",
        explanation: str | None = None,
    ) -> dict[str, Any]:
        row = self._item_lookup().get(str(item_id), {})
        return {
            "item_id": str(item_id),
            "title": str(row.get("title", "")),
            "category": str(row.get("category", "")),
            "brand": str(row.get("brand", "")),
            "price": float(row.get("price", 0.0) or 0.0),
            "avg_rating": float(row.get("avg_rating", 0.0) or 0.0),
            "rating_count": int(row.get("rating_count", 0) or 0),
            "score": float(score or 0.0),
            "rank": int(rank or 0),
            "source": source,
            "explanation": explanation,
        }

    def understand_query(self, query_text: str) -> dict[str, Any]:
        """Run Stage 8-style query understanding with the mock client."""
        items = self._items()
        extractor = LLMIntentExtractor(
            MockLLMClient(model="mock-query-understanding-v1"),
            known_categories=sorted(items["category"].dropna().astype(str).unique().tolist()),
            known_brands=sorted(items["brand"].dropna().astype(str).unique().tolist()),
            max_expanded_queries=3,
        )
        parsed = extractor.parse_query(query_text)
        return {
            "query_text": query_text,
            "parsed_query": parsed,
            "provider": str(parsed.get("provider", "mock")),
            "model": str(parsed.get("model", "mock-query-understanding-v1")),
            "latency_ms": float(parsed.get("latency_ms", 0.0) or 0.0),
        }

    def _candidate_rows_for_text(
        self,
        query_text: str,
        top_k: int,
        use_multimodal_candidates: bool,
    ) -> tuple[str | None, pd.DataFrame, bool]:
        pairs = self._query_pairs()
        exact = pairs[pairs["query_text"].astype(str).str.lower() == query_text.lower()]
        candidates = self._candidates(use_multimodal_candidates)
        if not exact.empty:
            query_id = str(exact.sort_values(["split", "query_id"]).iloc[0]["query_id"])
            return (
                query_id,
                select_top_candidates_for_query(candidates, query_id, top_k),
                False,
            )

        items = self._items().copy()
        query_tokens = set(tokenize(query_text))
        scores = []
        for _, row in items.iterrows():
            text_tokens = set(tokenize(build_item_text(row)))
            overlap = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
            scores.append(float(overlap))
        items["_score"] = scores
        items = items.sort_values(
            ["_score", "avg_rating", "item_id"],
            ascending=[False, False, True],
        )
        rows = pd.DataFrame(
            {
                "candidate_item_id": items["item_id"].astype(str).head(top_k),
                "fallback_score": items["_score"].head(top_k),
                "fallback_rank": list(range(1, min(top_k, len(items)) + 1)),
            }
        )
        return None, rows, True

    def search(
        self,
        query_text: str,
        top_k: int = 10,
        use_llm_query_understanding: bool = True,
        use_multimodal_candidates: bool = True,
        include_explanations: bool = True,
    ) -> dict[str, Any]:
        """Search existing candidates or fallback to simple local text matching."""
        parsed = self.understand_query(query_text) if use_llm_query_understanding else None
        query_id, rows, fallback_used = self._candidate_rows_for_text(
            query_text,
            top_k,
            use_multimodal_candidates,
        )
        score_col = self._score_column(rows) or "fallback_score"
        rank_col = "multimodal_rank" if "multimodal_rank" in rows.columns else "fallback_rank"
        source = (
            self._candidate_key(use_multimodal_candidates) if not fallback_used else "text_fallback"
        )
        items = []
        for index, row in rows.head(top_k).reset_index(drop=True).iterrows():
            item_id = str(row.get("candidate_item_id", row.get("item_id", "")))
            explanation = None
            if include_explanations:
                item = self._item_lookup().get(item_id, {})
                explanation = (
                    f"Matched local demo query using {item.get('category', 'catalog')} "
                    f"metadata from {item.get('brand', 'catalog')}."
                )
            items.append(
                self._enrich_item(
                    item_id,
                    score=float(row.get(score_col, 0.0) or 0.0),
                    rank=int(row.get(rank_col, index + 1) or index + 1),
                    source=source,
                    explanation=explanation,
                )
            )
        return {
            "query_text": query_text,
            "query_id": query_id,
            "parsed_query": parsed["parsed_query"] if parsed else None,
            "candidate_source": source,
            "fallback_used": fallback_used,
            "items": items,
            "message": (
                None if not fallback_used else "No exact query_id found; used text fallback."
            ),
        }

    def recommend(self, user_id: str, top_k: int = 10) -> dict[str, Any]:
        """Return a local user recommendation demo from profiles and metadata."""
        profiles = self._read_parquet("user_profiles_path", required=False)
        profile = None
        if not profiles.empty and "user_id" in profiles.columns:
            matches = profiles[profiles["user_id"].astype(str) == str(user_id)]
            if not matches.empty:
                profile = self._json_safe(matches.iloc[0].to_dict())
        items = self._items().copy()
        categories = set(str(value) for value in (profile or {}).get("top_categories", []) or [])
        brands = set(str(value) for value in (profile or {}).get("top_brands", []) or [])
        items["_profile_score"] = (
            items["category"].astype(str).isin(categories).astype(float)
            + items["brand"].astype(str).isin(brands).astype(float)
            + pd.to_numeric(items["avg_rating"], errors="coerce").fillna(0.0) / 5.0
            + pd.to_numeric(items["rating_count"], errors="coerce").fillna(0.0).map(math.log1p)
            / 10.0
        )
        items = items.sort_values(
            ["_profile_score", "rating_count", "item_id"],
            ascending=[False, False, True],
        )
        output = [
            self._enrich_item(
                str(row["item_id"]),
                score=float(row["_profile_score"]),
                rank=index + 1,
                source="profile_metadata_fallback",
            )
            for index, (_, row) in enumerate(items.head(top_k).iterrows())
        ]
        return {
            "user_id": str(user_id),
            "items": output,
            "fallback_used": True,
            "profile": profile,
            "message": (
                "Used local profile metadata fallback; no request-time recommender training "
                "was run."
                if profile
                else "User profile not found; used catalog popularity fallback."
            ),
        }

    def _context_for_genrec(
        self,
        query_text: str,
        query_id: str | None,
        candidate_pool_size: int,
    ) -> tuple[str | None, list[dict[str, Any]], pd.DataFrame]:
        candidates = self._candidates(True)
        pairs = self._query_pairs()
        resolved_query_id = query_id
        if resolved_query_id is None:
            exact = pairs[pairs["query_text"].astype(str).str.lower() == query_text.lower()]
            if not exact.empty:
                resolved_query_id = str(
                    exact.sort_values(["split", "query_id"]).iloc[0]["query_id"]
                )
        if resolved_query_id is not None:
            candidate_rows = select_top_candidates_for_query(
                candidates,
                resolved_query_id,
                candidate_pool_size,
            )
        else:
            _, candidate_rows, _ = self._candidate_rows_for_text(
                query_text,
                candidate_pool_size,
                True,
            )
        return (
            resolved_query_id,
            format_candidates_for_llm(
                candidate_rows,
                self._items(),
                max_candidates=candidate_pool_size,
            ),
            candidate_rows,
        )

    def genrec(
        self,
        query_text: str,
        query_id: str | None = None,
        top_k: int = 10,
        candidate_pool_size: int = 20,
        method: str = "candidate_constrained_generation",
    ) -> dict[str, Any]:
        """Return candidate-constrained GenRec output and diagnostics."""
        saved = self._read_parquet("genrec_outputs_path", required=False)
        if query_id and not saved.empty:
            matches = saved[
                (saved["query_id"].astype(str) == str(query_id))
                & (saved["method"].astype(str) == method)
            ]
            if not matches.empty:
                row = matches.iloc[0].to_dict()
                ranked_items = row.get("ranked_items", []) or []
                return self._genrec_response(query_text, query_id, row, ranked_items)

        resolved_query_id, candidates, _ = self._context_for_genrec(
            query_text,
            query_id,
            candidate_pool_size,
        )
        client = MockLLMClient(model="mock-genrec-v1")
        if method.startswith("llm_rerank_top_"):
            pool_size = int(method.rsplit("_", 1)[-1])
            output = LLMReranker(client, candidate_pool_size=pool_size).rerank(
                query_text,
                candidates,
                top_k=top_k,
            )
        else:
            output = CandidateConstrainedGenerator(client).generate(
                query_text,
                candidates,
                top_k=top_k,
            )
        ranked_items = output.get("ranked_items", [])
        return self._genrec_response(query_text, resolved_query_id, output, ranked_items)

    def _genrec_response(
        self,
        query_text: str,
        query_id: str | None,
        output: dict[str, Any],
        ranked_items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        valid_ids = set(self._items()["item_id"].astype(str))
        emitted_ids = [str(row.get("item_id", "")) for row in ranked_items]
        invalid = list(output.get("invalid_item_ids", []) or [])
        invalid.extend([item_id for item_id in emitted_ids if item_id and item_id not in valid_ids])
        invalid = list(dict.fromkeys(invalid))
        denominator = len(emitted_ids) or len(invalid)
        enriched = []
        for index, row in enumerate(ranked_items):
            item_id = str(row.get("item_id", ""))
            item = self._item_lookup().get(item_id, {})
            enriched.append(
                {
                    "item_id": item_id,
                    "rank": int(row.get("rank", index + 1) or index + 1),
                    "reason": str(row.get("reason", "")),
                    "confidence": float(row.get("confidence", 0.0) or 0.0),
                    "title": str(item.get("title", "")),
                    "category": str(item.get("category", "")),
                    "brand": str(item.get("brand", "")),
                }
            )
        explanations = []
        if emitted_ids:
            explanations = (
                TemplateExplanationGenerator()
                .generate(
                    query_id=query_id or "ad_hoc_query",
                    query_text=query_text,
                    recommended_item_ids=emitted_ids,
                    items=self._items(),
                )
                .get("explanations", [])
            )
        return {
            "query_text": query_text,
            "query_id": query_id,
            "method": str(output.get("method", "candidate_constrained_generation")),
            "provider": str(output.get("provider", "mock")),
            "model": str(output.get("model", "mock-genrec-v1")),
            "recommended_items": enriched,
            "invalid_item_ids": invalid,
            "valid_item_rate": (
                (len(emitted_ids) - len(invalid)) / denominator if denominator else 0.0
            ),
            "hallucination_rate": len(invalid) / denominator if denominator else 0.0,
            "used_fallback": bool(output.get("used_fallback", False)),
            "parse_success": bool(output.get("parse_success", True)),
            "schema_valid": bool(output.get("schema_valid", not invalid)),
            "explanations": explanations,
        }

    def model_comparison(self) -> dict[str, Any]:
        """Return final leaderboard and best method per stage."""
        leaderboard = self._read_csv("final_leaderboard_path")
        if leaderboard.empty:
            return {"leaderboard": [], "best_by_stage": [], "message": "Leaderboard is empty."}
        sorted_board = leaderboard.sort_values(
            [
                "stage",
                "primary_score",
                "secondary_score",
                "coverage_score",
                "avg_latency_ms",
                "method",
            ],
            ascending=[True, False, False, False, True, True],
        )
        best = sorted_board.groupby("stage", sort=True).head(1)
        return {
            "leaderboard": self._records(leaderboard),
            "best_by_stage": self._records(best),
            "message": "Metrics come from synthetic local debug data.",
        }

    def business_metrics(self) -> dict[str, Any]:
        """Summarize synthetic proxy business metrics from artifacts."""
        items = self._items()
        candidates = self._read_parquet("ranking_candidates_multimodal_path", required=False)
        multimodal = self._read_csv("genrec_results_path", required=False)
        metrics = {
            "num_items": int(len(items)),
            "num_categories": int(items["category"].nunique()) if "category" in items else 0,
            "avg_rating": float(pd.to_numeric(items["avg_rating"], errors="coerce").mean()),
            "avg_rating_count": float(pd.to_numeric(items["rating_count"], errors="coerce").mean()),
            "synthetic_data_caveat": "Synthetic proxy metrics only; not real GMV.",
        }
        if not candidates.empty:
            for column in ["authority_score", "conversion_proxy", "item_event_weight_sum"]:
                if column in candidates.columns:
                    metrics[f"avg_{column}"] = float(
                        pd.to_numeric(candidates[column], errors="coerce").mean()
                    )
        if not multimodal.empty and "hallucination_rate" in multimodal.columns:
            metrics["avg_genrec_hallucination_rate"] = float(
                pd.to_numeric(multimodal["hallucination_rate"], errors="coerce").mean()
            )
        return metrics

    def error_taxonomy(self) -> dict[str, Any]:
        """Return existing error taxonomy markdown."""
        text = self._read_text("error_taxonomy_path", required=False)
        return {"markdown": text, "available": bool(text)}


def command_hint_for_missing(path: str | Path) -> str:
    """Return a generic missing-artifact message for dashboards."""
    return artifact_missing_message(path)
