"""Jina AI embedding client with document-vector reuse for one session."""
import hashlib

import httpx
import numpy as np

from src.config import Config
from src.errors import AppError, provider_error


def normalized_vectors(values, expected_rows: int) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != expected_rows or matrix.shape[1] == 0:
            raise ValueError("Unexpected embedding shape")
        if not np.isfinite(matrix).all():
            raise ValueError("Non-finite embedding")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if (norms <= 0).any() or not np.isfinite(norms).all():
            raise ValueError("Invalid norm")
        return np.ascontiguousarray(matrix / norms, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AppError("Jina returned invalid embedding vectors.") from exc


class JinaEmbeddings:
    def __init__(self, config: Config, client=None):
        self.config, self.client = config, client
        self.cache: dict[str, np.ndarray] = {}
        self.dimension: int | None = None

    def _embed(self, texts: list[str], task: str) -> np.ndarray:
        if not texts or any(not text.strip() for text in texts):
            raise AppError("Cannot embed empty text.")
        if not self.config.jina_api_key and self.client is None:
            raise AppError("Add JINA_API_KEY to the server environment to index and search documents.")
        client = self.client or httpx.Client(timeout=45)
        try:
            response = client.post("https://api.jina.ai/v1/embeddings", headers={
                "Authorization": f"Bearer {self.config.jina_api_key}",
                "Content-Type": "application/json",
            }, json={"model": self.config.embedding_model, "task": task,
                    "input": texts, "embedding_type": "float"})
            response.raise_for_status()
            payload = response.json()
            data = sorted(payload["data"], key=lambda row: row["index"])
            vectors = normalized_vectors([row["embedding"] for row in data], len(texts))
        except AppError:
            raise
        except Exception as exc:
            raise provider_error("Jina AI embeddings", exc) from exc
        if self.dimension is not None and vectors.shape[1] != self.dimension:
            raise AppError("Jina changed vector dimensions. Start a new session.")
        self.dimension = vectors.shape[1]
        return vectors

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        keys = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        missing = {}
        for key, value in zip(keys, texts):
            if key not in self.cache:
                missing[key] = value
        items = list(missing.items())
        for start in range(0, len(items), self.config.batch_size):
            batch = items[start:start + self.config.batch_size]
            vectors = self._embed([text for _, text in batch], "retrieval.passage")
            for (key, _), vector in zip(batch, vectors):
                self.cache[key] = vector
        return np.ascontiguousarray(np.stack([self.cache[key] for key in keys]), dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed([query], "retrieval.query")

    def retain(self, texts: list[str]) -> None:
        keys = {hashlib.sha256(text.encode()).hexdigest() for text in texts}
        self.cache = {key: vector for key, vector in self.cache.items() if key in keys}
