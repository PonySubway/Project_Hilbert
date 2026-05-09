"""Render query responses as terminal table, Markdown, or JSON."""

from __future__ import annotations

import json
from dataclasses import asdict

from .schemas import QueryResponse, SearchResult


def render_response(response: QueryResponse, output_format: str = "table") -> str:
    """Render a response using table, json, or markdown format."""
    normalized = output_format.strip().lower()
    if normalized == "json":
        return render_json(response)
    if normalized in {"md", "markdown"}:
        return render_markdown(response)
    if normalized == "table":
        return render_table(response)
    raise ValueError("format must be one of: table, json, markdown")


def render_json(response: QueryResponse) -> str:
    """Render stable JSON."""
    return json.dumps(asdict(response), ensure_ascii=False, indent=2)


def render_markdown(response: QueryResponse) -> str:
    """Render GitHub-friendly Markdown."""
    lines = [
        f"## Query: {response.query}",
        "",
        f"Mode: `{response.mode}`  ",
        f"Model: `{response.model}`",
        "",
    ]
    for cluster_id, results in _group_results(response.results):
        title = "Unclustered" if cluster_id is None else f"Cluster {cluster_id}"
        lines.extend(
            [
                f"### {title}",
                "| Rank | Concept | Similarity | Distance | Salient |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for result in results:
            rank = result.metadata.get("rank", "-")
            lines.append(
                f"| {rank} | {result.concept} | {result.similarity:.4f} | "
                f"{result.distance:.4f} | {result.salient:.4f} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def render_table(response: QueryResponse) -> str:
    """Render a plain terminal table without third-party dependencies."""
    rows = []
    for rank, result in enumerate(response.results, start=1):
        cluster = "-" if result.cluster_id is None else str(result.cluster_id)
        rows.append(
            [
                str(rank),
                result.concept,
                f"{result.similarity:.4f}",
                f"{result.distance:.4f}",
                cluster,
                f"{result.salient:.4f}",
            ]
        )

    headers = ["#", "Concept", "Similarity", "Distance", "Cluster", "Salient"]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(row: list[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    separator = "  ".join("-" * width for width in widths)
    lines = [
        f"Query: {response.query}",
        f"Mode:  {response.mode}",
        f"Model: {response.model}",
        "",
        format_row(headers),
        separator,
    ]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def _group_results(results: list[SearchResult]) -> list[tuple[int | None, list[SearchResult]]]:
    groups: list[tuple[int | None, list[SearchResult]]] = []
    positions: dict[int | None, int] = {}
    for result in results:
        cluster_id = result.cluster_id
        if cluster_id not in positions:
            positions[cluster_id] = len(groups)
            groups.append((cluster_id, []))
        groups[positions[cluster_id]][1].append(result)
    return groups

