"""High-level Project Hilbert query engine."""

from __future__ import annotations

from .clustering import cluster_neighbors
from .embeddings import EmbeddingModel
from .index import VectorIndex
from .salience import compute_salience
from .schemas import QueryResponse, RawNeighbor, SearchResult
from .vector_logic import evaluate_expression


class HilbertEngine:
    """Coordinates embedding, indexing, clustering, and salience."""

    def __init__(self, index: VectorIndex, embedder: EmbeddingModel) -> None:
        self.index = index
        self.embedder = embedder

    def search(
        self,
        query: str,
        top_n: int = 12,
        cluster: bool = True,
        eps: float = 0.35,
        min_samples: int = 2,
    ) -> QueryResponse:
        """Search concepts near a plain text query."""
        query_vector = self.embedder.encode([query])[0]
        return self._search_vector(query, "search", query_vector, top_n, cluster, eps, min_samples)

    def logic(
        self,
        expression: str,
        top_n: int = 12,
        cluster: bool = True,
        eps: float = 0.35,
        min_samples: int = 2,
    ) -> QueryResponse:
        """Evaluate a vector-logic expression and search the result vector."""
        query_vector = evaluate_expression(expression, self.embedder)
        return self._search_vector(expression, "logic", query_vector, top_n, cluster, eps, min_samples)

    def _search_vector(
        self,
        query: str,
        mode: str,
        query_vector: list[float],
        top_n: int,
        cluster: bool,
        eps: float,
        min_samples: int,
    ) -> QueryResponse:
        neighbors = self.index.search(query_vector, top_n=top_n)
        return response_from_neighbors(query, mode, self.embedder.name, neighbors, query_vector, cluster, eps, min_samples)


def response_from_neighbors(
    query: str,
    mode: str,
    model: str,
    neighbors: list[RawNeighbor],
    query_vector: list[float],
    cluster: bool = True,
    eps: float = 0.35,
    min_samples: int = 2,
    display_n: int | None = None,
) -> QueryResponse:
    """Build a user-facing response from nearest-neighbor candidates."""
    vectors = [neighbor.vector for neighbor in neighbors]
    can_cluster = bool(vectors) and all(vector for vector in vectors)
    labels = cluster_neighbors(vectors, eps=eps, min_samples=min_samples) if cluster and can_cluster else [None for _ in vectors]
    salient_values = compute_salience(vectors, query_vector=query_vector) if can_cluster else [0.0 for _ in vectors]

    output_neighbors = neighbors if display_n is None else neighbors[:display_n]
    results = [
        SearchResult(
            concept=neighbor.concept,
            similarity=round(neighbor.similarity, 6),
            distance=round(neighbor.distance, 6),
            cluster_id=labels[index],
            salient=round(salient_values[index], 6),
            metadata={"rank": neighbor.rank, **neighbor.metadata},
        )
        for index, neighbor in enumerate(output_neighbors)
    ]
    return QueryResponse(query=query, mode=mode, model=model, results=results)

