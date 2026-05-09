"""Corpus and lexicon import utilities.

Project Hilbert cannot decode arbitrary vectors directly into words. It needs a
large lexical universe. This module creates that universe from public Chinese
word sources and user-provided text files.
"""

from __future__ import annotations

from pathlib import Path
import warnings


def normalize_term(term: str) -> str:
    """Normalize a raw term while preserving Chinese text."""
    return " ".join(term.strip().split())


def is_valid_term(term: str, min_len: int = 1, max_len: int = 64) -> bool:
    """Return whether a term should be kept in the lexicon."""
    if not term:
        return False
    if term.startswith("#"):
        return False
    if len(term) < min_len or len(term) > max_len:
        return False
    if any(char in term for char in "\r\n\t"):
        return False
    return True


def read_word_file(path: str | Path, first_column: bool = True) -> list[str]:
    """Read terms from a UTF-8-ish text file.

    Many public dictionaries use formats like `word frequency tag`; by default
    only the first whitespace-separated column is imported.
    """
    word_path = Path(path)
    text = _read_text_fallback(word_path)
    terms: list[str] = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        term = raw.split()[0] if first_column else raw
        terms.append(normalize_term(term))
    return terms


def write_clean_terms(
    terms: list[str],
    out: str | Path,
    min_len: int = 1,
    max_len: int = 64,
    limit: int = 0,
) -> int:
    """Deduplicate and write clean terms. Returns number of written terms."""
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    written = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for raw_term in terms:
            term = normalize_term(raw_term)
            if term in seen or not is_valid_term(term, min_len=min_len, max_len=max_len):
                continue
            seen.add(term)
            file.write(term + "\n")
            written += 1
            if limit > 0 and written >= limit:
                break
    return written


def import_word_files(
    inputs: list[str | Path],
    out: str | Path,
    min_len: int = 1,
    max_len: int = 64,
    limit: int = 0,
    first_column: bool = True,
) -> int:
    """Import one or more word files into a clean one-term-per-line lexicon."""
    terms: list[str] = []
    for input_path in inputs:
        terms.extend(read_word_file(input_path, first_column=first_column))
    return write_clean_terms(terms, out, min_len=min_len, max_len=max_len, limit=limit)


def import_jieba_dictionary(
    out: str | Path,
    min_len: int = 1,
    max_len: int = 64,
    limit: int = 0,
) -> int:
    """Import jieba's public Chinese dictionary into a clean lexicon file."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")
            import jieba  # type: ignore
    except ImportError as exc:
        raise RuntimeError("jieba is not installed. Run `pip install jieba`.") from exc

    dict_file = jieba.get_dict_file()
    terms = []
    for line in dict_file:
        raw = line.decode("utf-8", errors="ignore") if isinstance(line, bytes) else line
        raw = raw.strip()
        if not raw:
            continue
        terms.append(raw.split()[0])
    return write_clean_terms(terms, out, min_len=min_len, max_len=max_len, limit=limit)


def _read_text_fallback(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")

