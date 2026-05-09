"""Small dependency-free vector utilities.

The first public version intentionally avoids NumPy so it can run in a fresh
Python environment. The API stays simple enough to replace with NumPy later.
"""

from __future__ import annotations

import math

Vector = list[float]


def dot(left: Vector, right: Vector) -> float:
    """Return the dot product of two equal-length vectors."""
    return sum(a * b for a, b in zip(left, right, strict=True))


def norm(vector: Vector) -> float:
    """Return the Euclidean norm of a vector."""
    return math.sqrt(dot(vector, vector))


def normalize(vector: Vector) -> Vector:
    """Return a normalized vector, preserving zero vectors."""
    length = norm(vector)
    if length == 0.0:
        return [0.0 for _ in vector]
    return [value / length for value in vector]


def add(left: Vector, right: Vector) -> Vector:
    """Return left + right."""
    return [a + b for a, b in zip(left, right, strict=True)]


def subtract(left: Vector, right: Vector) -> Vector:
    """Return left - right."""
    return [a - b for a, b in zip(left, right, strict=True)]


def scale(vector: Vector, factor: float) -> Vector:
    """Return vector * factor."""
    return [value * factor for value in vector]


def cosine_similarity(left: Vector, right: Vector) -> float:
    """Return cosine similarity in [-1, 1]."""
    left_norm = norm(left)
    right_norm = norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return clamp(dot(left, right) / (left_norm * right_norm), -1.0, 1.0)


def cosine_distance(left: Vector, right: Vector) -> float:
    """Return cosine distance in [0, 2]."""
    return clamp(1.0 - cosine_similarity(left, right), 0.0, 2.0)


def mean(values: list[float]) -> float:
    """Return the arithmetic mean, or 0.0 for an empty list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Clamp a value into a closed interval."""
    return max(minimum, min(maximum, value))

