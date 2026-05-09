from unittest import TestCase

from project_hilbert.embeddings import SemanticMockEmbeddingModel
from project_hilbert.vector_logic import evaluate_expression, parse_expression
from project_hilbert.vector_math import cosine_similarity


class VectorLogicTests(TestCase):
    def test_parse_expression(self) -> None:
        self.assertEqual(parse_expression("北京 - 中国 + 英国"), ["北京", "-", "中国", "+", "英国"])

    def test_parse_rejects_empty_expression(self) -> None:
        with self.assertRaises(ValueError):
            parse_expression("   ")

    def test_parse_rejects_double_operator(self) -> None:
        with self.assertRaises(ValueError):
            parse_expression("北京 - + 英国")

    def test_capital_analogy_points_to_london(self) -> None:
        embedder = SemanticMockEmbeddingModel()
        result = evaluate_expression("北京 - 中国 + 英国", embedder)
        london = embedder.encode(["伦敦"])[0]
        china = embedder.encode(["中国"])[0]
        self.assertGreater(cosine_similarity(result, london), cosine_similarity(result, china))

    def test_royalty_analogy_points_to_queen(self) -> None:
        embedder = SemanticMockEmbeddingModel()
        result = evaluate_expression("国王 - 男人 + 女人", embedder)
        queen = embedder.encode(["女王"])[0]
        king = embedder.encode(["国王"])[0]
        self.assertGreater(cosine_similarity(result, queen), 0.75)
        self.assertLess(cosine_similarity(result, king), cosine_similarity(result, queen))

