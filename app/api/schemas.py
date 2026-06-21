"""Pydantic schemas for the local FastAPI demo."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ArtifactStatus(BaseModel):
    """Artifact existence status."""

    name: str
    path: str
    exists: bool
    required: bool = False
    command_hint: Optional[str] = None


class HealthResponse(BaseModel):
    """API health response."""

    name: str
    version: str
    environment: str
    status: str
    artifacts_available: int
    artifacts_total: int


class SearchRequest(BaseModel):
    """Search demo request."""

    query_text: str = Field(min_length=1)
    top_k: int = Field(default=10, gt=0, le=50)
    use_llm_query_understanding: bool = True
    use_multimodal_candidates: bool = True
    include_explanations: bool = True


class SearchResultItem(BaseModel):
    """Enriched search result item."""

    item_id: str
    title: str = ""
    category: str = ""
    brand: str = ""
    price: float = 0.0
    avg_rating: float = 0.0
    rating_count: int = 0
    score: float = 0.0
    rank: int = 0
    source: str = ""
    explanation: Optional[str] = None


class SearchResponse(BaseModel):
    """Search demo response."""

    query_text: str
    query_id: Optional[str] = None
    parsed_query: Optional[dict[str, Any]] = None
    candidate_source: str
    fallback_used: bool
    items: list[SearchResultItem]
    message: Optional[str] = None


class RecommendResponse(BaseModel):
    """User recommendation response."""

    user_id: str
    items: list[SearchResultItem]
    fallback_used: bool
    profile: Optional[dict[str, Any]] = None
    message: Optional[str] = None


class QueryUnderstandingRequest(BaseModel):
    """Query-understanding request."""

    query_text: str = Field(min_length=1)


class QueryUnderstandingResponse(BaseModel):
    """Query-understanding response."""

    query_text: str
    parsed_query: dict[str, Any]
    provider: str
    model: str
    latency_ms: float


class GenRecRequest(BaseModel):
    """Candidate-constrained GenRec request."""

    query_text: str = Field(min_length=1)
    query_id: Optional[str] = None
    top_k: int = Field(default=10, gt=0, le=50)
    candidate_pool_size: int = Field(default=20, gt=0, le=100)
    method: str = "candidate_constrained_generation"


class GenRecItem(BaseModel):
    """GenRec recommended item."""

    item_id: str
    rank: int
    reason: str = ""
    confidence: float = 0.0
    title: str = ""
    category: str = ""
    brand: str = ""


class GenRecResponse(BaseModel):
    """Candidate-constrained GenRec response."""

    query_text: str
    query_id: Optional[str] = None
    method: str
    provider: str
    model: str
    recommended_items: list[GenRecItem]
    invalid_item_ids: list[str]
    valid_item_rate: float
    hallucination_rate: float
    used_fallback: bool
    parse_success: bool
    schema_valid: bool
    explanations: list[dict[str, Any]] = Field(default_factory=list)


class ModelComparisonResponse(BaseModel):
    """Model comparison response."""

    leaderboard: list[dict[str, Any]]
    best_by_stage: list[dict[str, Any]]
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Structured error response."""

    error: str
    message: str
    command_hint: Optional[str] = None
