"""FastAPI routes for the TikSearchRec-LLM local demo."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from app.api.dependencies import get_demo_service
from app.api.schemas import (
    GenRecRequest,
    GenRecResponse,
    HealthResponse,
    ModelComparisonResponse,
    QueryUnderstandingRequest,
    QueryUnderstandingResponse,
    RecommendResponse,
    SearchRequest,
    SearchResponse,
)
from app.api.service import DemoService

router = APIRouter()
ServiceDep = Annotated[DemoService, Depends(get_demo_service)]


@router.get("/health", response_model=HealthResponse, tags=["status"])
def health(service: ServiceDep) -> dict[str, Any]:
    """Return service health."""
    return service.health()


@router.get("/artifacts/status", tags=["status"])
def artifact_status(service: ServiceDep) -> dict[str, Any]:
    """Return configured artifact status."""
    return service.artifact_status()


@router.post("/search", response_model=SearchResponse, tags=["demo"])
def search(
    request: SearchRequest,
    service: ServiceDep,
) -> dict[str, Any]:
    """Run a local search demo."""
    return service.search(
        query_text=request.query_text,
        top_k=request.top_k,
        use_llm_query_understanding=request.use_llm_query_understanding,
        use_multimodal_candidates=request.use_multimodal_candidates,
        include_explanations=request.include_explanations,
    )


@router.get("/recommend/{user_id}", response_model=RecommendResponse, tags=["demo"])
def recommend(
    user_id: str,
    service: ServiceDep,
    top_k: int = 10,
) -> dict[str, Any]:
    """Run a local user recommendation demo."""
    return service.recommend(user_id=user_id, top_k=top_k)


@router.post(
    "/llm/query-understanding",
    response_model=QueryUnderstandingResponse,
    tags=["llm"],
)
def understand_query(
    request: QueryUnderstandingRequest,
    service: ServiceDep,
) -> dict[str, Any]:
    """Run mock LLM query understanding."""
    return service.understand_query(request.query_text)


@router.post("/genrec", response_model=GenRecResponse, tags=["llm"])
def genrec(
    request: GenRecRequest,
    service: ServiceDep,
) -> dict[str, Any]:
    """Run candidate-constrained GenRec."""
    return service.genrec(
        query_text=request.query_text,
        query_id=request.query_id,
        top_k=request.top_k,
        candidate_pool_size=request.candidate_pool_size,
        method=request.method,
    )


@router.get("/models/leaderboard", response_model=ModelComparisonResponse, tags=["validation"])
def models_leaderboard(service: ServiceDep) -> dict[str, Any]:
    """Return the final validation leaderboard."""
    return service.model_comparison()


@router.get("/metrics/business", tags=["validation"])
def business_metrics(service: ServiceDep) -> dict[str, Any]:
    """Return local synthetic business/proxy metrics."""
    return service.business_metrics()


@router.get("/errors/taxonomy", tags=["validation"])
def error_taxonomy(service: ServiceDep) -> dict[str, Any]:
    """Return the error taxonomy markdown."""
    return service.error_taxonomy()
