"""Salient value estimation."""

from __future__ import annotations

from .vector_math import Vector, clamp, cosine_similarity, mean


def compute_salience(
    vectors: list[Vector],
    query_vector: Vector | None = None,
    k: int = 5,
    alpha: float = 0.70,
    beta: float = 0.30,
) -> list[float]:
    """Compute salient values in [0, 1] for a list of candidate vectors.

    The first version combines two intuitions:
    - sparse local neighborhoods should be more salient;
    - candidates should still remain related to the query when a query exists.
    """
    if not vectors:
        return []
    if len(vectors) == 1:
        return [1.0]
    if k <= 0:
        raise ValueError("k must be positive")

    densities = [_local_density(vectors, index, k) for index in range(len(vectors))]
    min_density = min(densities)
    max_density = max(densities)

    salience_scores: list[float] = []
    for vector, density in zip(vectors, densities, strict=True):
        if max_density == min_density:
            sparsity = 0.5
        else:
            sparsity = 1.0 - ((density - min_density) / (max_density - min_density))

        if query_vector is None:
            relatedness = 0.5
        else:
            relatedness = (cosine_similarity(query_vector, vector) + 1.0) / 2.0

        salience_scores.append(clamp(alpha * sparsity + beta * relatedness))
    return salience_scores


def _local_density(vectors: list[Vector], index: int, k: int) -> float:
    similarities = []
    for other_index, other_vector in enumerate(vectors):
        if other_index == index:
            continue
        similarities.append(cosine_similarity(vectors[index], other_vector))
    similarities.sort(reverse=True)
    return mean(similarities[:k])

