import io
import urllib.error
from unittest import TestCase

from project_hilbert.web_core import (
    DeepSeekAPIError,
    DeepSeekClient,
    DeepSeekConfigError,
    DeepSeekResponseError,
    HilbertWebService,
)


CONCEPTS = [
    "苹果",
    "香蕉",
    "梨",
    "橙子",
    "葡萄",
    "西瓜",
    "谷歌",
    "微软",
    "英伟达",
    "亚马逊",
    "特斯拉",
    "芯片",
    "人工智能",
    "智能手机",
    "北京",
    "中国",
    "伦敦",
    "英国",
    "国王",
    "女王",
    "男人",
    "女人",
]


class WebCoreTests(TestCase):
    def setUp(self) -> None:
        self.service = HilbertWebService.from_concepts(CONCEPTS)

    def test_ring_returns_at_least_five_stable_layers(self) -> None:
        response = self.service.ring("苹果", layers=5, per_layer=3)
        self.assertEqual(response["query"], "苹果")
        self.assertEqual(len(response["layers"]), 5)
        self.assertEqual([layer["layer"] for layer in response["layers"]], [1, 2, 3, 4, 5])
        self.assertGreater(response["total_results"], 0)
        self.assertNotIn("苹果", [item["concept"] for layer in response["layers"] for item in layer["items"]])

    def test_relation_score_is_clamped_to_zero_one(self) -> None:
        response = self.service.relate("苹果", "香蕉")
        self.assertGreaterEqual(response["score"], 0.0)
        self.assertLessEqual(response["score"], 1.0)
        self.assertGreaterEqual(response["inner_product"], -1.0)
        self.assertLessEqual(response["inner_product"], 1.0)

    def test_related_words_score_higher_than_unrelated_words(self) -> None:
        related = self.service.relate("苹果", "香蕉")
        unrelated = self.service.relate("苹果", "伦敦")
        self.assertGreater(related["score"], unrelated["score"])


class DeepSeekClientTests(TestCase):
    def test_explain_requires_api_key(self) -> None:
        client = DeepSeekClient(api_key=None)
        with self.assertRaises(DeepSeekConfigError):
            client.explain_word("苹果")

    def test_explain_extracts_content_from_valid_response(self) -> None:
        body = '{"choices":[{"message":{"content":"义项一"}}]}'.encode("utf-8")
        client = DeepSeekClient(api_key="test", urlopen=_fake_urlopen(body))
        response = client.explain_word("苹果", query="水果")
        self.assertEqual(response["explanation"], "义项一")
        self.assertEqual(response["model"], "deepseek-v4-flash")

    def test_explain_raises_api_error_for_http_error(self) -> None:
        client = DeepSeekClient(api_key="test", urlopen=_failing_urlopen)
        with self.assertRaises(DeepSeekAPIError) as context:
            client.explain_word("苹果")
        self.assertIn("bad request", str(context.exception))

    def test_explain_raises_response_error_for_invalid_json(self) -> None:
        client = DeepSeekClient(api_key="test", urlopen=_fake_urlopen(b"not json"))
        with self.assertRaises(DeepSeekResponseError):
            client.explain_word("苹果")

    def test_explain_raises_response_error_for_missing_content(self) -> None:
        client = DeepSeekClient(api_key="test", urlopen=_fake_urlopen(b'{"choices":[]}'))
        with self.assertRaises(DeepSeekResponseError):
            client.explain_word("苹果")


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def _fake_urlopen(body: bytes):
    def open_response(request, *, timeout: float):
        return _FakeResponse(body)

    return open_response


def _failing_urlopen(request, *, timeout: float):
    raise urllib.error.HTTPError(
        url="https://api.deepseek.com/chat/completions",
        code=400,
        msg="Bad Request",
        hdrs=None,
        fp=io.BytesIO(b'{"error":{"message":"bad request"}}'),
    )
