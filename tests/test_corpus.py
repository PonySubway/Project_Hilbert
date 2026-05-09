from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from project_hilbert.corpus import import_word_files, normalize_term, write_clean_terms


class CorpusTests(TestCase):
    def test_normalize_term_collapses_spaces(self) -> None:
        self.assertEqual(normalize_term("  人工   智能  "), "人工 智能")

    def test_write_clean_terms_deduplicates_and_filters(self) -> None:
        with TemporaryDirectory() as directory:
            out = Path(directory) / "terms.txt"
            count = write_clean_terms(["苹果", "", "苹果", "香蕉", "#comment"], out)
            self.assertEqual(count, 2)
            self.assertEqual(out.read_text(encoding="utf-8").splitlines(), ["苹果", "香蕉"])

    def test_import_word_files_uses_first_column(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "dict.txt"
            out = Path(directory) / "terms.txt"
            source.write_text("苹果 100 n\n香蕉 80 n\n", encoding="utf-8")
            count = import_word_files([source], out)
            self.assertEqual(count, 2)
            self.assertEqual(out.read_text(encoding="utf-8").splitlines(), ["苹果", "香蕉"])

