"""FastAPI app factory for TikSearchRec-LLM."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.dependencies import get_config
from app.api.errors import ArtifactMissingError, InvalidDemoRequestError
from app.api.routes import router


def create_app() -> FastAPI:
    """Create the FastAPI application without starting a server."""
    config = get_config()
    app_config = config.get("app", {})
    app = FastAPI(
        title=str(app_config.get("name", "TikSearchRec-LLM API")),
        version=str(app_config.get("version", "0.11.0")),
        description="Local artifact-backed TikSearchRec-LLM demo API.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8501",
            "http://127.0.0.1:8501",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ArtifactMissingError)
    async def artifact_missing_handler(
        request: Request,
        exc: ArtifactMissingError,
    ) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=404,
            content={
                "error": "artifact_missing",
                "message": str(exc),
                "command_hint": exc.command_hint,
            },
        )

    @app.exception_handler(InvalidDemoRequestError)
    async def invalid_request_handler(
        request: Request,
        exc: InvalidDemoRequestError,
    ) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_demo_request", "message": str(exc)},
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        del request, exc
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": "Unexpected local demo error. Check server logs for details.",
            },
        )

    @app.get("/", tags=["status"])
    def root() -> dict[str, str]:
        return {"message": "TikSearchRec-LLM API", "docs": "/docs", "health": "/health"}

    app.include_router(router)
    return app


app = create_app()
