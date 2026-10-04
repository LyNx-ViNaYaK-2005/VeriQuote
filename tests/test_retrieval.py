import unittest
from unittest.mock import Mock

import numpy as np

from src.config import Config
from src.errors import AppError
from src.models import Chunk
from src.retrieval.embeddings import SentenceTransformerEmbeddings
from src.retrieval.retriever import retrieve
from src.retrieval.vector_store import VectorStore


def chunk(identity, text, page=1):
    return Chunk(identity, "doc", text, "study.pdf", page, 0)


class RetrievalTests(unittest.TestCase):
    def test_local_embeddings_work_with_faiss_inner_product(self):
        # Non-unit vectors and a non-default dimension verify normalization
        # and dynamic dimensions through the local adapter and real FAISS.
        model = Mock()
        model.encode.side_effect = [
            np.array([[2, 0, 0, 0, 0], [0, 3, 0, 0, 0]], dtype=np.float64),
            np.array([[0, 10, 0, 0, 0]], dtype=np.float64),
        ]
        embedder = SentenceTransformerEmbeddings(Config(), model)
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
        for call in model.encode.call_args_list:
            self.assertIs(call.kwargs["normalize_embeddings"], True)
            self.assertIs(call.kwargs["convert_to_numpy"], True)

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

    def test_mismatched_vectors(self):
        with self.assertRaises(AppError):
            VectorStore([chunk("a", "hello")], np.array([[1, 0], [0, 1]]))
        store = VectorStore([chunk("a", "hello")], np.array([[1, 0]]))
        with self.assertRaises(AppError):
            store.search(np.array([[1, 0, 0]]), 2)
