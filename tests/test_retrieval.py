import unittest
from unittest.mock import Mock

import numpy as np

from src.config import Config
from src.errors import AppError
from src.models import Chunk
from src.retrieval.embeddings import JinaEmbeddings
from src.retrieval.retriever import retrieve
from src.retrieval.vector_store import VectorStore


def chunk(identity, text, page=1):
    return Chunk(identity, "doc", text, "study.pdf", page, 0)


class RetrievalTests(unittest.TestCase):
    def test_jina_vectors_work_with_faiss_inner_product(self):
        from fakes import FakeJinaClient
        client = FakeJinaClient()
        # API-shaped deterministic vectors keep this test offline.
        client.post = Mock(side_effect=[
            type("Response", (), {"raise_for_status": lambda self: None, "json": lambda self: {"data": [
                {"index": 0, "embedding": [2, 0, 0, 0, 0]}, {"index": 1, "embedding": [0, 3, 0, 0, 0]}]}})(),
            type("Response", (), {"raise_for_status": lambda self: None, "json": lambda self: {"data": [
                {"index": 0, "embedding": [0, 10, 0, 0, 0]}]}})(),
        ])
        embedder = JinaEmbeddings(Config(jina_api_key="test"), client)
        chunks = [chunk("a", "Apples are fruit."), chunk("b", "Networks transport packets.", 3)]
        vectors = embedder.embed_documents([item.text for item in chunks])
        query = embedder.embed_query("How are packets transported?")
        self.assertEqual(vectors.dtype, np.float32)
        self.assertEqual(query.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1)
        np.testing.assert_allclose(np.linalg.norm(query, axis=1), 1)
        store = VectorStore(chunks, vectors)
        self.assertEqual(store.index.d, 5)
        hits = retrieve(store, query, min_score=0.25)
        self.assertEqual([hit.chunk for hit in hits], [chunks[1]])
        self.assertAlmostEqual(hits[0].score, 1)

    def test_faiss_vector_mapping_and_score(self):
        chunks = [chunk("a", "Apples are fruit."), chunk("b", "Networks transport packets.", 3)]
        store = VectorStore(chunks, np.array([[1, 0], [0, 1]]))
        hits = store.search(np.array([[0, 10]]), 20)
        self.assertEqual(hits[0].chunk, chunks[1])
        self.assertAlmostEqual(hits[0].score, 1)
        self.assertEqual(hits[0].chunk.page, 3)

    def test_repetition_threshold_and_budget(self):
        chunks = [chunk("a", "A protocol transports packets across the network."),
                  chunk("b", "A protocol transports packets across the network."),
                  chunk("c", "Another unrelated idea about apples.")]
        store = VectorStore(chunks, np.array([[1, 0], [1, 0], [0, 1]]))
        hits = retrieve(store, np.array([[1, 0]]), min_score=0.2)
        self.assertEqual(len(hits), 1)
        self.assertEqual(retrieve(store, np.array([[-1, 0]]), min_score=0.2), [])
        self.assertEqual(retrieve(store, np.array([[1, 0]]), max_chars=10), [])

    def test_positive_top_match_survives_low_absolute_score(self):
        chunks = [chunk("relevant", "The report describes evidence-based retrieval."),
                  chunk("less", "A separate passage covers unrelated procedures.", 2)]
        vectors = np.array([[0.20, np.sqrt(0.96)], [0.10, np.sqrt(0.99)]], dtype=np.float32)
        store = VectorStore(chunks, vectors)
        hits = retrieve(store, np.array([[1.0, 0.0]]), min_score=0.25)
        self.assertEqual(hits[0].chunk, chunks[0])
        self.assertAlmostEqual(hits[0].score, 0.20, places=5)

    def test_broad_retrieval_covers_distinct_pages_without_similarity_cutoff(self):
        chunks = [chunk("a", "Page one contains a substantive research finding.", 1),
                  chunk("b", "Page two explains the method and its limits.", 2),
                  chunk("c", "Page three presents implications for future work.", 3)]
        store = VectorStore(chunks, np.array([[0.1, np.sqrt(.99)], [0.2, np.sqrt(.96)],
                                               [0.15, np.sqrt(.9775)]]))
        hits = retrieve(store, np.array([[1.0, 0.0]]), top_k=3,
                        min_score=0.25, broad=True, query_text="Summarize this document")
        self.assertEqual({hit.chunk.page for hit in hits}, {1, 2, 3})

    def test_mismatched_vectors(self):
        with self.assertRaises(AppError):
            VectorStore([chunk("a", "hello")], np.array([[1, 0], [0, 1]]))
        store = VectorStore([chunk("a", "hello")], np.array([[1, 0]]))
        with self.assertRaises(AppError):
            store.search(np.array([[1, 0, 0]]), 2)
