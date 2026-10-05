import unittest
from types import SimpleNamespace

import numpy as np

from src.config import Config
from src.errors import AppError
from src.retrieval.embeddings import JinaEmbeddings, normalized_vectors


class FakeJinaClient:
    def __init__(self, vectors=None):
        self.calls = []
        self.vectors = vectors

    def post(self, url, headers, json):
        self.calls.append(json)
        values = self.vectors or [[1.0, 0.2, 0.1] for _ in json["input"]]
        return SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {"data": [{"index": i, "embedding": vector}
                                                       for i, vector in enumerate(values)]})


class EmbeddingTests(unittest.TestCase):
    def test_batched_jina_request_cache_query_and_retain(self):
        client = FakeJinaClient()
        embedder = JinaEmbeddings(Config(jina_api_key="test"), client)
        vectors = embedder.embed_documents(["a", "b", "c", "a"])
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["model"], "jina-embeddings-v3")
        self.assertEqual(client.calls[0]["task"], "retrieval.passage")
        self.assertEqual(client.calls[0]["input"], ["a", "b", "c"])
        self.assertEqual(vectors.shape, (4, 3))
        self.assertEqual(vectors.dtype, np.float32)
        embedder.embed_documents(["a"])
        self.assertEqual(len(client.calls), 1)
        query = embedder.embed_query("question")
        self.assertEqual(client.calls[-1]["task"], "retrieval.query")
        self.assertEqual(query.shape, (1, 3))
        embedder.retain(["a"])
        self.assertEqual(len(embedder.cache), 1)

    def test_missing_key_empty_text_and_bad_vectors(self):
        with self.assertRaisesRegex(AppError, "JINA_API_KEY"):
            JinaEmbeddings(Config()).embed_documents(["document"])
        embedder = JinaEmbeddings(Config(jina_api_key="test"), FakeJinaClient())
        with self.assertRaisesRegex(AppError, "empty text"):
            embedder.embed_query(" ")
        for value in ([[0, 0]], [[float("nan"), 1]], [[1, 2], [3, 4]]):
            with self.assertRaises(AppError):
                normalized_vectors(value, 1)


if __name__ == "__main__":
    unittest.main()
