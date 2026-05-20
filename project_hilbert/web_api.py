"""FastAPI application for Project Hilbert's web UI."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .web_core import (
    DeepSeekAPIError,
    DeepSeekConfigError,
    DeepSeekResponseError,
    HilbertWebService,
)


class RingRequest(BaseModel):
    """Request body for semantic ring search."""

    query: str = Field(min_length=1)
    layers: int = Field(default=5, ge=5, le=12)
    per_layer: int = Field(default=8, ge=1, le=24)


class ExplainRequest(BaseModel):
    """Request body for word explanation."""

    word: str = Field(min_length=1)
    query: str = ""


class RelateRequest(BaseModel):
    """Request body for relation scoring."""

    left: str = Field(min_length=1)
    right: str = Field(min_length=1)


@lru_cache(maxsize=1)
def get_service() -> HilbertWebService:
    """Return the singleton web service."""
    return HilbertWebService.from_env()


def create_app(service_factory: Callable[[], HilbertWebService] = get_service) -> FastAPI:
    """Create the FastAPI app."""
    app = FastAPI(
        title="Project Hilbert Web API",
        version="0.1.0",
        description="Semantic ring search, word explanations, and relation scoring.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict:
        return service_factory().health()

    @app.post("/api/ring")
    def ring(request: RingRequest) -> dict:
        try:
            return service_factory().ring(
                request.query,
                layers=request.layers,
                per_layer=request.per_layer,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/explain")
    def explain(request: ExplainRequest) -> dict:
        try:
            return service_factory().explain(request.word, query=request.query)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except DeepSeekConfigError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (DeepSeekAPIError, DeepSeekResponseError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/api/relate")
    def relate(request: RelateRequest) -> dict:
        try:
            return service_factory().relate(request.left, request.right)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
