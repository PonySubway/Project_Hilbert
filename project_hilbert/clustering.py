"""Dependency-free DBSCAN clustering for nearest-neighbor vectors."""

from __future__ import annotations

from .vector_math import Vector, cosine_distance

NOISE = -1
UNVISITED = -99


def cluster_neighbors(
    vectors: list[Vector],
    eps: float = 0.35,
    min_samples: int = 2,
) -> list[int | None]:
    """Cluster vectors using a tiny DBSCAN implementation.

    Returns None for noise points and for cases where no meaningful cluster is
    discovered.
    """
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if len(vectors) < min_samples:
        return [None for _ in vectors]

    labels = [UNVISITED for _ in vectors]
    cluster_id = 0

    for point_index in range(len(vectors)):
        if labels[point_index] != UNVISITED:
            continue
        neighbors = _region_query(vectors, point_index, eps)
        if len(neighbors) < min_samples:
            labels[point_index] = NOISE
            continue
        _expand_cluster(vectors, labels, point_index, neighbors, cluster_id, eps, min_samples)
        cluster_id += 1

    mapped = [None if label == NOISE else int(label) for label in labels]
    if all(label is None for label in mapped):
        return [None for _ in vectors]
    return mapped


def _region_query(vectors: list[Vector], point_index: int, eps: float) -> list[int]:
    return [
        other_index
        for other_index, other_vector in enumerate(vectors)
        if cosine_distance(vectors[point_index], other_vector) <= eps
    ]


def _expand_cluster(
    vectors: list[Vector],
    labels: list[int],
    point_index: int,
    neighbors: list[int],
    cluster_id: int,
    eps: float,
    min_samples: int,
) -> None:
    labels[point_index] = cluster_id
    queue = list(neighbors)
    cursor = 0
    while cursor < len(queue):
        neighbor_index = queue[cursor]
        cursor += 1

        if labels[neighbor_index] == NOISE:
            labels[neighbor_index] = cluster_id
        if labels[neighbor_index] != UNVISITED:
            continue

        labels[neighbor_index] = cluster_id
        neighbor_neighbors = _region_query(vectors, neighbor_index, eps)
        if len(neighbor_neighbors) >= min_samples:
            for candidate in neighbor_neighbors:
                if candidate not in queue:
                    queue.append(candidate)

