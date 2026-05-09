"""Embedding model adapters.

Version 0.1 ships with a deterministic semantic mock model so the project can
run offline and without heavy dependencies. It also includes a lazy
sentence-transformers adapter for future use with models such as BAAI/bge-m3.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from .vector_math import Vector, add, normalize, scale


class EmbeddingModel(Protocol):
    """Protocol implemented by all embedding models."""

    name: str
    dimensions: int

    def encode(self, texts: list[str]) -> list[Vector]:
        """Encode texts into normalized vectors."""


FEATURES: tuple[str, ...] = (
    "fruit",
    "technology_company",
    "hardware",
    "ai",
    "city",
    "country",
    "capital_role",
    "europe",
    "asia",
    "royalty",
    "male",
    "female",
    "human",
    "profession",
    "institution",
    "medicine",
    "education",
    "vehicle",
    "phone",
    "commerce",
    "language",
    "finance",
    "culture",
    "science",
    "china_id",
    "uk_id",
    "france_id",
    "japan_id",
)

FEATURE_INDEX = {name: index for index, name in enumerate(FEATURES)}
DEFAULT_DIMENSIONS = 64


CONCEPT_FEATURES: dict[str, dict[str, float]] = {
    # Fruits and Apple ambiguity.
    "苹果": {"fruit": 1.0, "technology_company": 0.75, "phone": 0.45, "hardware": 0.25},
    "apple": {"fruit": 1.0, "technology_company": 0.75, "phone": 0.45, "hardware": 0.25},
    "香蕉": {"fruit": 1.0},
    "banana": {"fruit": 1.0},
    "梨": {"fruit": 1.0},
    "橙子": {"fruit": 1.0},
    "葡萄": {"fruit": 1.0},
    "西瓜": {"fruit": 1.0},
    # Technology companies and artifacts.
    "谷歌": {"technology_company": 1.0, "ai": 0.7, "commerce": 0.2},
    "google": {"technology_company": 1.0, "ai": 0.7, "commerce": 0.2},
    "微软": {"technology_company": 1.0, "ai": 0.55},
    "microsoft": {"technology_company": 1.0, "ai": 0.55},
    "英伟达": {"technology_company": 1.0, "hardware": 0.95, "ai": 0.95},
    "nvidia": {"technology_company": 1.0, "hardware": 0.95, "ai": 0.95},
    "亚马逊": {"technology_company": 0.95, "commerce": 0.95},
    "amazon": {"technology_company": 0.95, "commerce": 0.95},
    "特斯拉": {"technology_company": 0.9, "vehicle": 1.0, "hardware": 0.4},
    "tesla": {"technology_company": 0.9, "vehicle": 1.0, "hardware": 0.4},
    "芯片": {"hardware": 1.0, "ai": 0.5, "technology_company": 0.2},
    "人工智能": {"ai": 1.0, "science": 0.4},
    "智能手机": {"phone": 1.0, "hardware": 0.7, "technology_company": 0.35},
    # Countries, cities, and capital relation.
    "中国": {"country": 1.0, "asia": 0.8, "culture": 0.4, "china_id": 1.0},
    "china": {"country": 1.0, "asia": 0.8, "culture": 0.4, "china_id": 1.0},
    "北京": {"country": 1.0, "asia": 0.8, "capital_role": 1.0, "city": 1.0, "culture": 0.5, "china_id": 1.0},
    "beijing": {"country": 1.0, "asia": 0.8, "capital_role": 1.0, "city": 1.0, "culture": 0.5, "china_id": 1.0},
    "英国": {"country": 1.0, "europe": 0.85, "culture": 0.45, "uk_id": 1.0},
    "uk": {"country": 1.0, "europe": 0.85, "culture": 0.45, "uk_id": 1.0},
    "united kingdom": {"country": 1.0, "europe": 0.85, "culture": 0.45, "uk_id": 1.0},
    "伦敦": {"country": 1.0, "europe": 0.85, "capital_role": 1.0, "city": 1.0, "finance": 0.6, "culture": 0.5, "uk_id": 1.0},
    "london": {"country": 1.0, "europe": 0.85, "capital_role": 1.0, "city": 1.0, "finance": 0.6, "culture": 0.5, "uk_id": 1.0},
    "法国": {"country": 1.0, "europe": 0.9, "culture": 0.7, "france_id": 1.0},
    "巴黎": {"country": 1.0, "europe": 0.9, "capital_role": 1.0, "city": 1.0, "culture": 0.9, "france_id": 1.0},
    "日本": {"country": 1.0, "asia": 0.9, "culture": 0.7, "japan_id": 1.0},
    "东京": {"country": 1.0, "asia": 0.9, "capital_role": 1.0, "city": 1.0, "technology_company": 0.15, "japan_id": 1.0},
    # Gender and royalty.
    "国王": {"royalty": 1.0, "male": 1.0, "human": 0.8},
    "king": {"royalty": 1.0, "male": 1.0, "human": 0.8},
    "女王": {"royalty": 1.0, "female": 1.0, "human": 0.8},
    "queen": {"royalty": 1.0, "female": 1.0, "human": 0.8},
    "男人": {"male": 1.0, "human": 1.0},
    "man": {"male": 1.0, "human": 1.0},
    "女人": {"female": 1.0, "human": 1.0},
    "woman": {"female": 1.0, "human": 1.0},
    # Professions and institutions.
    "医生": {"profession": 1.0, "medicine": 1.0, "human": 0.6},
    "医院": {"institution": 1.0, "medicine": 1.0},
    "教师": {"profession": 1.0, "education": 1.0, "human": 0.6},
    "学校": {"institution": 1.0, "education": 1.0},
}


class SemanticMockEmbeddingModel:
    """A deterministic local embedding model for demos and tests.

    It is not a replacement for a real neural embedding model. Its purpose is to
    make Project Hilbert useful immediately in a terminal and to keep tests
    offline. Known concepts get interpretable semantic features; unknown texts
    receive deterministic hashed features.
    """

    name = "semantic-mock"

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS) -> None:
        if dimensions < len(FEATURES) + 8:
            raise ValueError(f"dimensions must be at least {len(FEATURES) + 8}")
        self.dimensions = dimensions

    def encode(self, texts: list[str]) -> list[Vector]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> Vector:
        key = text.strip().lower()
        vector = [0.0 for _ in range(self.dimensions)]

        feature_values = CONCEPT_FEATURES.get(key)
        if feature_values is None:
            feature_values = self._infer_features(key)

        for feature, value in feature_values.items():
            index = FEATURE_INDEX.get(feature)
            if index is not None:
                vector[index] += value

        vector = add(vector, scale(self._hash_vector(key), 0.08))
        return normalize(vector)

    def _infer_features(self, key: str) -> dict[str, float]:
        features: dict[str, float] = {}
        if any(word in key for word in ("公司", "科技", "ai", "智能", "tech")):
            features["technology_company"] = 0.65
        if any(word in key for word in ("城市", "city", "京", "都")):
            features["city"] = 0.55
        if any(word in key for word in ("国", "country")):
            features["country"] = 0.55
        if any(word in key for word in ("水果", "果", "fruit")):
            features["fruit"] = 0.65
        if not features:
            features["language"] = 0.25
        return features

    def _hash_vector(self, key: str) -> Vector:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        values: Vector = [0.0 for _ in range(self.dimensions)]
        start = len(FEATURES)
        for offset, byte in enumerate(digest):
            index = start + (offset % (self.dimensions - start))
            values[index] += (byte / 255.0) * 2.0 - 1.0
        return normalize(values)


class SentenceTransformerEmbeddingModel:
    """Lazy adapter for sentence-transformers models such as BAAI/bge-m3."""

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        try:
            # noinspection PyPackageRequirements
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install optional dependencies "
                "with `pip install sentence-transformers`, or use --model mock."
            ) from exc

        self.name = model_name
        self._model = SentenceTransformer(model_name)
        if hasattr(self._model, "get_embedding_dimension"):
            self.dimensions = int(self._model.get_embedding_dimension())
        else:
            self.dimensions = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> list[Vector]:
        vectors = self._model.encode(
            texts,
            convert_to_numpy=False,
            normalize_embeddings=True,
        )
        return [[float(value) for value in vector] for vector in vectors]


def create_embedding_model(model: str) -> EmbeddingModel:
    """Create an embedding model by CLI-friendly name."""
    normalized = model.strip().lower()
    if normalized in {"mock", "semantic-mock", "local"}:
        return SemanticMockEmbeddingModel()
    if normalized in {"mini", "minilm"}:
        return SentenceTransformerEmbeddingModel(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    if normalized in {"bge-m3", "baai/bge-m3"}:
        local_bge_m3 = Path(__file__).resolve().parent.parent / "models" / "bge-m3"
        if local_bge_m3.exists():
            return SentenceTransformerEmbeddingModel(str(local_bge_m3))
        return SentenceTransformerEmbeddingModel("BAAI/bge-m3")
    return SentenceTransformerEmbeddingModel(model)

