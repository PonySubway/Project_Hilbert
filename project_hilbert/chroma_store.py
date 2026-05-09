"""Chroma-backed vector store for large lexical universes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .embeddings import EmbeddingModel
from .index import load_concepts
from .schemas import RawNeighbor
from .vector_math import Vector, clamp

DEFAULT_CHROMA_DIR = Path(".chroma")
DEFAULT_COLLECTION = "project_hilbert_zh"


def stable_id(text: str) -> str:
    """Return a stable Chroma id for a term."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"term-{digest}"


class ChromaVectorStore:
    """Persistent Chroma store used for vector -> word nearest-neighbor lookup."""

    def __init__(self, persist_dir: str | Path = DEFAULT_CHROMA_DIR, collection: str = DEFAULT_COLLECTION) -> None:
        try:
            import chromadb  # type: ignore
        except ImportError as exc:
            raise RuntimeError("chromadb is not installed. Run `pip install chromadb`.") from exc

        self.persist_dir = Path(persist_dir)
        self.collection_name = collection
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine", "project": "Project Hilbert"},
        )

    def reset(self) -> None:
        """Delete and recreate the collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "project": "Project Hilbert"},
        )

    def count(self) -> int:
        """Return the number of stored concepts."""
        return int(self.collection.count())

    def build_from_file(
        self,
        concepts_path: str | Path,
        embedder: EmbeddingModel,
        batch_size: int = 64,
        source: str = "lexicon",
        reset: bool = False,
        limit: int = 0,
    ) -> int:
        """Embed a concept file and upsert it into Chroma."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if reset:
            self.reset()

        concepts = load_concepts(concepts_path)
        if limit > 0:
            concepts = concepts[:limit]

        inserted = 0
        for start in range(0, len(concepts), batch_size):
            batch = concepts[start : start + batch_size]
            embeddings = embedder.encode(batch)
            ids = [stable_id(text) for text in batch]
            metadatas = [
                {
                    "text": text,
                    "source": source,
                    "model": embedder.name,
                    "rank_in_source": start + offset,
                }
                for offset, text in enumerate(batch)
            ]
            self.collection.upsert(
                ids=ids,
                documents=batch,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            inserted += len(batch)
            print(f"indexed {inserted}/{len(concepts)} terms into {self.collection_name}")
        return inserted

    def search_vector(self, query_vector: Vector, top_n: int = 30) -> list[RawNeighbor]:
        """Search Chroma by an already computed query vector."""
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        result = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_n,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        documents = _first(result.get("documents"), [])
        metadatas = _first(result.get("metadatas"), [])
        distances = _first(result.get("distances"), [])
        embeddings = _first(result.get("embeddings"), [])

        neighbors: list[RawNeighbor] = []
        for index, document in enumerate(documents):
            distance = clamp(float(distances[index]) if index < len(distances) else 1.0, 0.0, 2.0)
            similarity = clamp(1.0 - distance, -1.0, 1.0)
            metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
            vector = _to_vector(embeddings[index]) if index < len(embeddings) else []
            neighbors.append(
                RawNeighbor(
                    concept=str(document),
                    vector=vector,
                    similarity=similarity,
                    distance=distance,
                    rank=index + 1,
                    metadata=metadata,
                )
            )
        return neighbors


def _first(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if len(value) == 0:
        return default
    return value[0]


def _to_vector(value: Any) -> Vector:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return [float(item) for item in value.tolist()]
    return [float(item) for item in value]

