"""Data structures used by Project Hilbert."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .vector_math import Vector


@dataclass(frozen=True)
class ConceptRecord:
    """A concept stored in the local concept library."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RawNeighbor:
    """A nearest-neighbor candidate before clustering and salience."""

    concept: str
    vector: Vector
    similarity: float
    distance: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """A complete search result returned to the user."""

    concept: str
    similarity: float
    distance: float
    cluster_id: int | None
    salient: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryResponse:
    """The top-level response for search and vector-logic queries."""

    query: str
    mode: str
    model: str
    results: list[SearchResult]

