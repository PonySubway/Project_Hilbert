"""Vector logic expression parsing and evaluation."""

from __future__ import annotations

import re

from .embeddings import EmbeddingModel
from .vector_math import Vector, add, normalize, subtract

OPERATORS = {"+", "-"}
TOKEN_PATTERN = re.compile(r"\s*([+-])\s*|\s*([^+-]+?)\s*(?=[+-]|$)")


def parse_expression(expression: str) -> list[str]:
    """Parse a left-to-right + / - vector expression.

    Examples:
    - 北京 - 中国 + 英国 -> ["北京", "-", "中国", "+", "英国"]
    - king - man + woman -> ["king", "-", "man", "+", "woman"]
    """
    if not expression or not expression.strip():
        raise ValueError("expression cannot be empty")

    tokens: list[str] = []
    position = 0
    while position < len(expression):
        match = TOKEN_PATTERN.match(expression, position)
        if match is None:
            raise ValueError(f"invalid expression near: {expression[position:]!r}")
        operator, concept = match.groups()
        token = operator if operator is not None else concept.strip()
        if token:
            tokens.append(token)
        position = match.end()

    _validate_tokens(tokens)
    return tokens


def _validate_tokens(tokens: list[str]) -> None:
    if not tokens:
        raise ValueError("expression cannot be empty")
    if tokens[0] in OPERATORS:
        raise ValueError("expression must start with a concept")
    if tokens[-1] in OPERATORS:
        raise ValueError("expression must end with a concept")
    for left, right in zip(tokens, tokens[1:], strict=False):
        if left in OPERATORS and right in OPERATORS:
            raise ValueError("two operators cannot appear next to each other")
        if left not in OPERATORS and right not in OPERATORS:
            raise ValueError("two concepts must be separated by an operator")


def evaluate_expression(expression: str, embedder: EmbeddingModel) -> Vector:
    """Evaluate a + / - expression into a normalized vector."""
    tokens = parse_expression(expression)
    result: Vector | None = None
    current_operator = "+"

    for token in tokens:
        if token in OPERATORS:
            current_operator = token
            continue

        vector = embedder.encode([token])[0]
        if result is None:
            result = vector
        elif current_operator == "+":
            result = add(result, vector)
        elif current_operator == "-":
            result = subtract(result, vector)
        else:  # pragma: no cover - impossible after validation
            raise ValueError(f"unsupported operator: {current_operator}")

    if result is None:  # pragma: no cover - validation prevents this
        raise ValueError("expression produced no vector")
    return normalize(result)

