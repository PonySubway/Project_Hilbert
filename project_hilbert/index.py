"""In-memory vector index."""

from __future__ import annotations

from pathlib import Path

from .embeddings import EmbeddingModel
from .schemas import RawNeighbor
from .vector_math import Vector, cosine_distance, cosine_similarity


class VectorIndex:
    """A small in-memory index suitable for the first terminal version."""

    def __init__(self, embedder: EmbeddingModel) -> None:
        self.embedder = embedder
        self.concepts: list[str] = []
        self.vectors: list[Vector] = []

    def build(self, concepts: list[str]) -> None:
        """Build the index from a list of concept texts."""
        cleaned = []
        seen = set()
        for concept in concepts:
            text = concept.strip()
            if not text or text.startswith("#") or text in seen:
                continue
            seen.add(text)
            cleaned.append(text)
        self.concepts = cleaned
        self.vectors = self.embedder.encode(cleaned)

    def build_from_file(self, path: str | Path) -> None:
        """Load one concept per line from a UTF-8 text file and build the index."""
        concept_path = Path(path)
        concepts = concept_path.read_text(encoding="utf-8").splitlines()
        self.build(concepts)

    def search(self, query_vector: Vector, top_n: int = 30) -> list[RawNeighbor]:
        """Search nearest concepts by cosine similarity."""
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        scored = []
        for concept, vector in zip(self.concepts, self.vectors, strict=True):
            similarity = cosine_similarity(query_vector, vector)
            scored.append((similarity, concept, vector))
        scored.sort(key=lambda item: item[0], reverse=True)
        neighbors: list[RawNeighbor] = []
        for rank, (similarity, concept, vector) in enumerate(scored[:top_n], start=1):
            neighbors.append(
                RawNeighbor(
                    concept=concept,
                    vector=vector,
                    similarity=similarity,
                    distance=cosine_distance(query_vector, vector),
                    rank=rank,
                )
            )
        return neighbors


def load_concepts(path: str | Path) -> list[str]:
    """Load a concept file without building an index."""
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

