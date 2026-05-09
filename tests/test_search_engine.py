from unittest import TestCase

from project_hilbert.embeddings import SemanticMockEmbeddingModel
from project_hilbert.engine import HilbertEngine
from project_hilbert.index import VectorIndex


class SearchEngineTests(TestCase):
    def setUp(self) -> None:
        self.embedder = SemanticMockEmbeddingModel()
        self.index = VectorIndex(self.embedder)
        self.index.build([
            "苹果",
            "香蕉",
            "梨",
            "谷歌",
            "英伟达",
            "北京",
            "中国",
            "伦敦",
            "英国",
            "国王",
            "女王",
            "男人",
            "女人",
        ])
        self.engine = HilbertEngine(self.index, self.embedder)

    def test_search_returns_ranked_results(self) -> None:
        response = self.engine.search("苹果", top_n=5)
        concepts = [result.concept for result in response.results]
        self.assertIn("苹果", concepts)
        self.assertTrue(any(concept in concepts for concept in ["香蕉", "梨"]))
        self.assertTrue(any(concept in concepts for concept in ["谷歌", "英伟达"]))

    def test_logic_returns_london_for_capital_expression(self) -> None:
        response = self.engine.logic("北京 - 中国 + 英国", top_n=3, cluster=False)
        concepts = [result.concept for result in response.results]
        self.assertIn("伦敦", concepts)

    def test_results_include_salience_and_cluster_fields(self) -> None:
        response = self.engine.search("苹果", top_n=6)
        for result in response.results:
            self.assertGreaterEqual(result.salient, 0.0)
            self.assertLessEqual(result.salient, 1.0)
            self.assertTrue(result.cluster_id is None or isinstance(result.cluster_id, int))

