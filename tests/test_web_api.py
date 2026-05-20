import importlib.util
import unittest


if importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("httpx") is None:
    raise unittest.SkipTest("FastAPI route tests require fastapi and httpx")

from fastapi.testclient import TestClient

from project_hilbert.web_api import create_app


class FakeService:
    def health(self):
        return {
            "status": "ok",
            "model": "semantic-mock",
            "configured_model": "mock",
            "search_backend": "memory",
            "concept_count": 2,
            "deepseek_configured": False,
        }

    def ring(self, query, layers=5, per_layer=8):
        return {"query": query, "model": "semantic-mock", "layers": [], "total_results": 0}

    def relate(self, left, right):
        return {
            "left": left,
            "right": right,
            "model": "semantic-mock",
            "inner_product": 0.5,
            "score": 0.5,
            "label": "中等相关",
        }

    def explain(self, word, query=None):
        return {"word": word, "query": query or "", "model": "deepseek-v4-flash", "explanation": "解释"}


class WebApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app(lambda: FakeService()))

    def test_health_route(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_relate_route(self) -> None:
        response = self.client.post("/api/relate", json={"left": "苹果", "right": "香蕉"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["score"], 0.5)
