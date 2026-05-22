"""Core services for the Project Hilbert web API."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .chroma_store import DEFAULT_CHROMA_DIR, ChromaVectorStore
from .embeddings import EmbeddingModel, create_embedding_model
from .engine import HilbertEngine, response_from_neighbors
from .index import VectorIndex
from .schemas import QueryResponse, SearchResult
from .vector_math import Vector, clamp, dot

DEFAULT_CONCEPTS = Path(__file__).resolve().parent.parent / "data" / "concepts.zh.txt"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class UrlOpen(Protocol):
    """Callable shape for urllib.request.urlopen."""

    def __call__(self, request: urllib.request.Request, *, timeout: float) -> Any:
        """Open a prepared request."""


@dataclass(frozen=True)
class WebSettings:
    """Runtime settings for the web service."""

    model: str = "mock"
    concepts: Path = DEFAULT_CONCEPTS
    collection: str | None = None
    persist_dir: Path = DEFAULT_CHROMA_DIR
    deepseek_api_key: str | None = None
    deepseek_model: str = DEFAULT_DEEPSEEK_MODEL
    deepseek_base_url: str = DEFAULT_DEEPSEEK_BASE_URL

    @classmethod
    def from_env(cls) -> "WebSettings":
        """Build settings from environment variables and an optional .env file."""
        values = load_env_settings()
        return cls(
            model=values.get("HILBERT_MODEL", "mock"),
            concepts=Path(values.get("HILBERT_CONCEPTS", str(DEFAULT_CONCEPTS))),
            collection=_blank_to_none(values.get("HILBERT_COLLECTION")),
            persist_dir=Path(values.get("HILBERT_PERSIST_DIR", str(DEFAULT_CHROMA_DIR))),
            deepseek_api_key=_blank_to_none(values.get("DEEPSEEK_API_KEY")),
            deepseek_model=values.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
            deepseek_base_url=values.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
        )


@dataclass(frozen=True)
class RingItem:
    """A positioned concept in the ring visualization."""

    concept: str
    similarity: float
    distance: float
    cluster_id: int | None
    salient: float
    rank: int


@dataclass(frozen=True)
class RingLayer:
    """One semantic distance layer in the ring visualization."""

    layer: int
    radius: float
    items: list[RingItem] = field(default_factory=list)


class DeepSeekConfigError(RuntimeError):
    """Raised when DeepSeek is not configured."""


class DeepSeekAPIError(RuntimeError):
    """Raised when DeepSeek returns an HTTP or service error."""


class DeepSeekResponseError(RuntimeError):
    """Raised when DeepSeek returns an unexpected response shape."""


@dataclass
class DeepSeekClient:
    """Small standard-library client for DeepSeek's OpenAI-compatible chat API."""

    api_key: str | None
    model: str = DEFAULT_DEEPSEEK_MODEL
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    timeout: float = 30.0
    urlopen: UrlOpen = urllib.request.urlopen

    @property
    def is_configured(self) -> bool:
        """Return whether the client has an API key."""
        return bool(self.api_key)

    def explain_word(self, word: str, query: str | None = None) -> dict[str, str]:
        """Ask DeepSeek to explain the meanings of a word."""
        clean_word = word.strip()
        clean_query = (query or "").strip()
        if not clean_word:
            raise ValueError("word must not be empty")
        if not self.api_key:
            raise DeepSeekConfigError("DEEPSEEK_API_KEY is not configured")

        user_prompt = (
            f"请解释词语“{clean_word}”可能具有的不同意思。"
            "输出 3 到 5 条编号解释，每条包含义项、简短说明和一个使用语境。"
        )
        if clean_query:
            user_prompt += f" 如果它和查询词“{clean_query}”有关，也说明这种关联。"
        messages = [
            {
                "role": "system",
                "content": "你是中文语义空间解释助手。回答要准确、简洁，避免编造无法确认的事实。",
            },
            {"role": "user", "content": user_prompt},
        ]
        explanation = self._chat(messages)
        return {
            "word": clean_word,
            "query": clean_query,
            "model": self.model,
            "explanation": explanation,
        }

    def _chat(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        response = self._post_json("/chat/completions", payload)
        return extract_chat_content(response)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}{path}",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with self.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raw_error = exc.read().decode("utf-8", errors="replace")
            raise DeepSeekAPIError(_format_deepseek_error(exc.code, raw_error)) from exc
        except urllib.error.URLError as exc:
            raise DeepSeekAPIError(f"DeepSeek request failed: {exc.reason}") from exc
        return parse_json_response(raw)


class HilbertWebService:
    """Application service used by both FastAPI and tests."""

    def __init__(
        self,
        settings: WebSettings,
        embedder: EmbeddingModel,
        index: VectorIndex | None = None,
        store: ChromaVectorStore | None = None,
        deepseek_client: DeepSeekClient | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.index = index
        self.store = store
        self.engine = HilbertEngine(index, embedder) if index is not None else None
        self.deepseek_client = deepseek_client or DeepSeekClient(
            api_key=settings.deepseek_api_key,
            model=settings.deepseek_model,
            base_url=settings.deepseek_base_url,
        )

    @classmethod
    def from_env(cls) -> "HilbertWebService":
        """Create a service from environment variables."""
        return cls.from_settings(WebSettings.from_env())

    @classmethod
    def from_settings(cls, settings: WebSettings) -> "HilbertWebService":
        """Create a service from explicit settings."""
        embedder = create_embedding_model(settings.model)
        if settings.collection:
            store = ChromaVectorStore(settings.persist_dir, settings.collection)
            return cls(settings, embedder, store=store)

        if not settings.concepts.exists():
            raise FileNotFoundError(f"concept file does not exist: {settings.concepts}")
        index = VectorIndex(embedder)
        index.build_from_file(settings.concepts)
        return cls(settings, embedder, index=index)

    @classmethod
    def from_concepts(cls, concepts: list[str], model: str = "mock") -> "HilbertWebService":
        """Create an in-memory service for tests."""
        settings = WebSettings(model=model)
        embedder = create_embedding_model(model)
        index = VectorIndex(embedder)
        index.build(concepts)
        return cls(settings, embedder, index=index)

    def health(self) -> dict[str, Any]:
        """Return runtime health information."""
        if self.store is not None:
            backend = "chroma"
            concept_count = self.store.count()
        else:
            backend = "memory"
            concept_count = len(self.index.concepts) if self.index is not None else 0
        return {
            "status": "ok",
            "model": self.embedder.name,
            "configured_model": self.settings.model,
            "search_backend": backend,
            "concept_count": concept_count,
            "deepseek_configured": self.deepseek_client.is_configured,
        }

    def ring(self, query: str, layers: int = 5, per_layer: int = 8) -> dict[str, Any]:
        """Return nearest concepts grouped into concentric semantic layers."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("query must not be empty")
        layer_count = max(5, layers)
        item_count = max(1, per_layer)
        response = self._search(clean_query, top_n=layer_count * item_count + 1)
        ring_layers = build_ring_layers(response, clean_query, layer_count, item_count)
        return {
            "query": clean_query,
            "model": response.model,
            "layers": [layer_to_dict(layer) for layer in ring_layers],
            "total_results": sum(len(layer.items) for layer in ring_layers),
        }

    def relate(self, left: str, right: str) -> dict[str, Any]:
        """Score the semantic relation between two terms."""
        clean_left = left.strip()
        clean_right = right.strip()
        if not clean_left or not clean_right:
            raise ValueError("left and right must not be empty")
        left_vector, right_vector = self.embedder.encode([clean_left, clean_right])
        inner_product = dot(left_vector, right_vector)
        score = clamp(inner_product, 0.0, 1.0)
        return {
            "left": clean_left,
            "right": clean_right,
            "model": self.embedder.name,
            "inner_product": round(inner_product, 6),
            "score": round(score, 6),
            "label": relation_label(score),
        }

    def explain(self, word: str, query: str | None = None) -> dict[str, str]:
        """Explain a word through DeepSeek."""
        return self.deepseek_client.explain_word(word, query=query)

    def _search(self, query: str, top_n: int) -> QueryResponse:
        if self.store is not None:
            query_vector = self.embedder.encode([query])[0]
            neighbors = self.store.search_vector(query_vector, top_n=top_n)
            return response_from_neighbors(query, "search", self.embedder.name, neighbors, query_vector)
        if self.engine is None:
            raise RuntimeError("no search backend is configured")
        return self.engine.search(query, top_n=top_n)


def build_ring_layers(response: QueryResponse, query: str, layers: int, per_layer: int) -> list[RingLayer]:
    """Split ranked results into stable concentric layers."""
    clean_query = query.strip().lower()
    filtered = [
        result
        for result in response.results
        if result.concept.strip().lower() != clean_query
    ]
    ring_layers: list[RingLayer] = []
    for layer_index in range(layers):
        start = layer_index * per_layer
        stop = start + per_layer
        items = [result_to_ring_item(result) for result in filtered[start:stop]]
        radius = round((layer_index + 1) / layers, 6)
        ring_layers.append(RingLayer(layer=layer_index + 1, radius=radius, items=items))
    return ring_layers


def result_to_ring_item(result: SearchResult) -> RingItem:
    """Convert a search result to ring output."""
    rank = result.metadata.get("rank", 0)
    return RingItem(
        concept=result.concept,
        similarity=result.similarity,
        distance=result.distance,
        cluster_id=result.cluster_id,
        salient=result.salient,
        rank=int(rank) if isinstance(rank, (int, float)) else 0,
    )


def layer_to_dict(layer: RingLayer) -> dict[str, Any]:
    """Convert a ring layer into JSON-friendly data."""
    return {
        "layer": layer.layer,
        "radius": layer.radius,
        "items": [asdict(item) for item in layer.items],
    }


def relation_label(score: float) -> str:
    """Return a compact Chinese label for a relation score."""
    if score >= 0.78:
        return "强相关"
    if score >= 0.45:
        return "中等相关"
    if score >= 0.18:
        return "弱相关"
    return "几乎无关"


def load_env_settings(env_file: str | Path | None = None, environ: dict[str, str] | None = None) -> dict[str, str]:
    """Load supported settings, with process environment taking precedence."""
    source_environ = os.environ if environ is None else environ
    path = Path(source_environ.get("HILBERT_ENV_FILE", str(env_file or DEFAULT_ENV_FILE)))
    values = parse_env_file(path) if path.exists() else {}
    for name in SUPPORTED_ENV_NAMES:
        if name in source_environ:
            values[name] = source_environ[name]
    return values


SUPPORTED_ENV_NAMES = {
    "HILBERT_MODEL",
    "HILBERT_CONCEPTS",
    "HILBERT_COLLECTION",
    "HILBERT_PERSIST_DIR",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_BASE_URL",
}


def parse_env_file(path: str | Path) -> dict[str, str]:
    """Parse a small dotenv file without adding a runtime dependency."""
    values: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()
        if "=" not in stripped:
            continue
        name, raw_value = stripped.split("=", 1)
        name = name.strip()
        if name not in SUPPORTED_ENV_NAMES:
            continue
        values[name] = _parse_env_value(raw_value.strip())
    return values


def _parse_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def parse_json_response(raw: str) -> dict[str, Any]:
    """Parse a JSON response body from DeepSeek."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DeepSeekResponseError("DeepSeek returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise DeepSeekResponseError("DeepSeek returned a non-object JSON payload")
    if "error" in payload:
        raise DeepSeekAPIError(_message_from_error_payload(payload))
    return payload


def extract_chat_content(payload: dict[str, Any]) -> str:
    """Extract choices[0].message.content from a chat completion payload."""
    try:
        choices = payload["choices"]
        message = choices[0]["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise DeepSeekResponseError("DeepSeek response did not contain message content") from exc
    if not isinstance(content, str) or not content.strip():
        raise DeepSeekResponseError("DeepSeek response message content was empty")
    return content.strip()


def _format_deepseek_error(status_code: int, raw: str) -> str:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        detail = raw.strip() or "empty error response"
    else:
        detail = _message_from_error_payload(payload)
    return f"DeepSeek API returned HTTP {status_code}: {detail}"


def _message_from_error_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("type") or error.get("code")
            if message:
                return str(message)
        if isinstance(error, str):
            return error
        message = payload.get("message")
        if message:
            return str(message)
    return "unknown DeepSeek error"


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
