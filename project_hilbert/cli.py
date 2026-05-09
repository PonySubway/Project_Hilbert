"""Command-line interface for Project Hilbert."""

from __future__ import annotations

import argparse
from pathlib import Path

from .chroma_store import DEFAULT_CHROMA_DIR, DEFAULT_COLLECTION, ChromaVectorStore
from .corpus import import_jieba_dictionary, import_word_files
from .embeddings import create_embedding_model
from .engine import HilbertEngine, response_from_neighbors
from .index import VectorIndex
from .renderers import render_response
from .vector_logic import evaluate_expression

DEFAULT_CONCEPTS = Path(__file__).resolve().parent.parent / "data" / "concepts.zh.txt"


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser."""
    parser = argparse.ArgumentParser(
        prog="project-hilbert",
        description="Explore concepts in a local vector space.",
    )
    _add_runtime_options(parser, use_defaults=True)

    subparsers = parser.add_subparsers(dest="command", required=True)

    search = subparsers.add_parser("search", help="Search concepts near a query.")
    search.add_argument("query", help="Text query, e.g. 苹果")
    _add_runtime_options(search, use_defaults=False)
    _add_chroma_query_options(search)
    _add_common_query_options(search)

    logic = subparsers.add_parser("logic", help="Evaluate vector logic, e.g. 北京 - 中国 + 英国.")
    logic.add_argument("expression", help="Expression using + and - operators.")
    _add_runtime_options(logic, use_defaults=False)
    _add_chroma_query_options(logic)
    _add_common_query_options(logic)

    import_words = subparsers.add_parser("import-words", help="Clean and merge public/user word files.")
    import_words.add_argument("inputs", nargs="+", help="Input word files. One term per line or first whitespace column.")
    import_words.add_argument("--out", default="corpus/zh_terms.txt", help="Output clean lexicon file.")
    import_words.add_argument("--min-len", type=int, default=1, help="Minimum term length.")
    import_words.add_argument("--max-len", type=int, default=64, help="Maximum term length.")
    import_words.add_argument("--limit", type=int, default=0, help="Maximum terms to write; 0 means no limit.")
    import_words.add_argument("--keep-line", action="store_true", help="Keep full line instead of first column.")

    import_jieba = subparsers.add_parser("import-jieba", help="Import jieba's public Chinese dictionary.")
    import_jieba.add_argument("--out", default="corpus/jieba_terms.txt", help="Output clean lexicon file.")
    import_jieba.add_argument("--min-len", type=int, default=1, help="Minimum term length.")
    import_jieba.add_argument("--max-len", type=int, default=64, help="Maximum term length.")
    import_jieba.add_argument("--limit", type=int, default=0, help="Maximum terms to write; 0 means no limit.")

    build_chroma = subparsers.add_parser("build-chroma", help="Embed a lexicon and persist it into Chroma.")
    build_chroma.add_argument("--model", default="mock", help="Embedding model used to build the collection.")
    build_chroma.add_argument("--concepts", default="corpus/zh_terms.txt", help="Clean lexicon file to index.")
    build_chroma.add_argument("--persist-dir", default=str(DEFAULT_CHROMA_DIR), help="Chroma persistent directory.")
    build_chroma.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name.")
    build_chroma.add_argument("--batch-size", type=int, default=64, help="Embedding/upsert batch size.")
    build_chroma.add_argument("--source", default="public_zh_lexicon", help="Metadata source label.")
    build_chroma.add_argument("--limit", type=int, default=0, help="Maximum terms to index; 0 means all.")
    build_chroma.add_argument("--reset", action="store_true", help="Delete and rebuild the collection first.")

    subparsers.add_parser("models", help="Show supported model aliases.")
    return parser


def _add_runtime_options(parser: argparse.ArgumentParser, use_defaults: bool) -> None:
    default = None if use_defaults else argparse.SUPPRESS
    parser.add_argument(
        "--model",
        default="mock" if use_defaults else default,
        help="Embedding model: mock, mini, bge-m3, or any sentence-transformers model name.",
    )
    parser.add_argument(
        "--concepts",
        default=str(DEFAULT_CONCEPTS) if use_defaults else default,
        help="UTF-8 concept file, one concept per line.",
    )


def _add_common_query_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--top-n", type=int, default=12, help="Number of results to return.")
    parser.add_argument(
        "--format",
        choices=("table", "json", "markdown"),
        default="table",
        help="Output format.",
    )
    parser.add_argument("--no-cluster", action="store_true", help="Disable DBSCAN clustering.")
    parser.add_argument("--eps", type=float, default=0.35, help="DBSCAN cosine-distance epsilon.")
    parser.add_argument("--min-samples", type=int, default=2, help="DBSCAN min_samples.")


def _add_chroma_query_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--persist-dir", default=str(DEFAULT_CHROMA_DIR), help="Chroma persistent directory.")
    parser.add_argument("--collection", default=None, help="Search a Chroma collection instead of --concepts.")
    parser.add_argument("--search-k", type=int, default=0, help="Internal Chroma candidates for clustering/salience; defaults to top-n.")


def main(argv: list[str] | None = None) -> int:
    """Run the command-line application."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "models":
        print(_models_text())
        return 0

    if args.command == "import-words":
        count = import_word_files(
            args.inputs,
            args.out,
            min_len=args.min_len,
            max_len=args.max_len,
            limit=args.limit,
            first_column=not args.keep_line,
        )
        print(f"wrote {count} clean terms to {args.out}")
        return 0

    if args.command == "import-jieba":
        count = import_jieba_dictionary(args.out, min_len=args.min_len, max_len=args.max_len, limit=args.limit)
        print(f"wrote {count} jieba terms to {args.out}")
        return 0

    if args.command == "build-chroma":
        embedder = create_embedding_model(args.model)
        store = ChromaVectorStore(args.persist_dir, args.collection)
        count = store.build_from_file(
            args.concepts,
            embedder,
            batch_size=args.batch_size,
            source=args.source,
            reset=args.reset,
            limit=args.limit,
        )
        print(f"collection={args.collection} count={store.count()} indexed_now={count}")
        return 0

    embedder = create_embedding_model(args.model)

    if getattr(args, "collection", None):
        store = ChromaVectorStore(args.persist_dir, args.collection)
        if args.command == "search":
            query_vector = embedder.encode([args.query])[0]
            search_k = args.search_k or args.top_n
            neighbors = store.search_vector(query_vector, top_n=search_k)
            response = response_from_neighbors(
                args.query,
                "search",
                embedder.name,
                neighbors,
                query_vector,
                cluster=not args.no_cluster,
                eps=args.eps,
                min_samples=args.min_samples,
                display_n=args.top_n,
            )
        elif args.command == "logic":
            query_vector = evaluate_expression(args.expression, embedder)
            search_k = args.search_k or args.top_n
            neighbors = store.search_vector(query_vector, top_n=search_k)
            response = response_from_neighbors(
                args.expression,
                "logic",
                embedder.name,
                neighbors,
                query_vector,
                cluster=not args.no_cluster,
                eps=args.eps,
                min_samples=args.min_samples,
                display_n=args.top_n,
            )
        else:  # pragma: no cover - argparse prevents this
            parser.error(f"unknown command: {args.command}")
        print(render_response(response, args.format))
        return 0

    concept_path = Path(args.concepts)
    if not concept_path.exists():
        parser.error(f"concept file does not exist: {concept_path}")

    index = VectorIndex(embedder)
    index.build_from_file(concept_path)
    engine = HilbertEngine(index, embedder)

    if args.command == "search":
        response = engine.search(
            args.query,
            top_n=args.top_n,
            cluster=not args.no_cluster,
            eps=args.eps,
            min_samples=args.min_samples,
        )
    elif args.command == "logic":
        response = engine.logic(
            args.expression,
            top_n=args.top_n,
            cluster=not args.no_cluster,
            eps=args.eps,
            min_samples=args.min_samples,
        )
    else:  # pragma: no cover - argparse prevents this
        parser.error(f"unknown command: {args.command}")

    print(render_response(response, args.format))
    return 0


def _models_text() -> str:
    return """Supported model aliases:

  mock      Offline deterministic semantic mock embedding. Default for v0.1.
  mini      sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2.
  bge-m3    BAAI/bge-m3. Downloads to local Hugging Face cache on first use.

Any other --model value is passed to sentence-transformers as a model name.
For tests and first-run demos, use --model mock.
""".strip()

